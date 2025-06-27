import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from topology import TrafficComplex
from topomodelx.nn.combinatorial.hmc import HMC
import json
import os
import time

# --- DATASET ---
class TrafficPredictionDataset(Dataset):
    def __init__(self, readings, input_window, pred_window=12, start=0, end=None):
        self.readings = readings
        self.input_window = input_window
        self.pred_window = pred_window
        self.num_sensors, self.num_timesteps = readings.shape
        self.starts = np.arange(start, (self.num_timesteps if end is None else end) - input_window - pred_window + 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        x = self.readings[:, start : start + self.input_window]
        y = self.readings[:, start + self.input_window : start + self.input_window + self.pred_window]
        return torch.tensor(x.T, dtype=torch.float32), torch.tensor(y.T, dtype=torch.float32)
    @staticmethod
    def last_window(readings, input_window, pred_window):
        num_timesteps = readings.shape[1]
        start = num_timesteps - input_window - pred_window
        return TrafficPredictionDataset(readings, input_window, pred_window, start=start, end=start+1)

def compute_metrics(y_true, y_pred):
    if hasattr(y_true, "detach"): y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"): y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mask = y_true > 1e-3  # Only use positive ground-truth speeds for MAPE
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    return rmse, mae, mape

def evaluate_model(model, dataloader, device, node_features, edge_features, face_features, edge_map, face_map, means, stds):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(
                x,
                node_features,
                edge_features,
                face_features,
                edge_map,
                face_map,
            )
            # Denormalize
            pred = pred.cpu().numpy() * stds[None, None, :] + means[None, None, :]
            y = y.cpu().numpy() * stds[None, None, :] + means[None, None, :]
            all_preds.append(pred)
            all_targets.append(y)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    print("y_true min/max/mean:", np.min(y_true), np.max(y_true), np.mean(y_true))
    print("y_pred min/max/mean:", np.min(y_pred), np.max(y_pred), np.mean(y_pred))
    print("num zeros in y_true:", np.sum(np.abs(y_true) < 1e-3), "/", y_true.size)
    print("num negatives in y_true:", np.sum(y_true < 0))
    print("num NaNs in y_true:", np.sum(np.isnan(y_true)))
    return compute_metrics(y_true, y_pred)

class TrafficCCNN(nn.Module):
    def __init__(self, complex, num_sensors, input_window, pred_window=12, hidden_dim=64, 
                 node_feat_dim=7, edge_feat_dim=4, face_feat_dim=3):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_window = input_window
        self.pred_window = pred_window
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Linear(hidden_dim + node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim + edge_feat_dim, hidden_dim)
        self.face_proj = nn.Linear(hidden_dim + face_feat_dim, hidden_dim)

        # HMC layers
        self.hmc1 = HMC([
            [
                [hidden_dim, hidden_dim, hidden_dim],  # 0-cells
                [hidden_dim, hidden_dim, hidden_dim],  # 1-cells
                [hidden_dim, hidden_dim, hidden_dim],  # 2-cells (faces)
            ]
        ], negative_slope=0.2)
        self.hmc2 = HMC([
            [
                [hidden_dim, hidden_dim, hidden_dim],  # 2-cells
                [hidden_dim, hidden_dim, hidden_dim],  # 1-cells
                [hidden_dim, hidden_dim, hidden_dim],  # 0-cells
            ]
        ], negative_slope=0.2)

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_window)
        )

        # Prepare adjacency/incidence matrices as buffers
        a0 = torch.from_numpy(complex.complex.adjacency_matrix(0, 1).todense()).to_sparse().float()
        a1 = torch.from_numpy(complex.complex.adjacency_matrix(1, 2).todense()).to_sparse().float()
        coa2_np = (complex.complex.incidence_matrix(0, 2).T @ complex.complex.incidence_matrix(0, 2))
        coa2_np.setdiag(0)
        coa2 = torch.from_numpy(coa2_np.todense()).to_sparse().float()
        b1 = torch.from_numpy(complex.complex.incidence_matrix(0, 1).todense()).to_sparse().float()
        b2 = torch.from_numpy(complex.complex.incidence_matrix(1, 2).todense()).to_sparse().float()
        self.register_buffer("a0", a0)
        self.register_buffer("a1", a1)
        self.register_buffer("coa2", coa2)
        self.register_buffer("b1", b1)
        self.register_buffer("b2", b2)

    def forward(self, x, node_features, edge_features, face_features, edge_map, face_map):
        batch, win, sensors = x.shape
        assert sensors == self.num_sensors

        # Temporal encoding for all sensors in the whole batch
        x_conv = x.permute(0, 2, 1).unsqueeze(2)  # shape: [batch, sensors, 1, time]
        x_conv = x_conv.reshape(-1, 1, x.shape[1])  # [batch * sensors, 1, time]
        h = self.temporal_encoder(x_conv)  # [batch * sensors, hidden_dim, time]
        h = h.mean(dim=2)  # [batch * sensors, hidden_dim]
        h = h.reshape(x.shape[0], x.shape[2], self.hidden_dim)  # [batch, sensors, hidden_dim]

        # Project node features for all batch entries at once
        node_input = torch.cat([
            h, 
            node_features.unsqueeze(0).expand(batch, -1, -1)
        ], dim=2)
        node_input = self.node_proj(node_input)

        # Project edge features (shared for all batch)
        edge_input = torch.cat([
            torch.zeros(edge_features.shape[0], self.hidden_dim, device=x.device),
            edge_features
        ], dim=1)
        edge_input = self.edge_proj(edge_input)

        # Project face features (shared for all batch)
        face_input = torch.cat([
            torch.zeros(face_features.shape[0], self.hidden_dim, device=x.device),
            face_features
        ], dim=1)
        face_input = self.face_proj(face_input)

        outputs = []
        for i in range(batch):
            x_0_out, x_1_out, x_2_out = self.hmc1(
                node_input[i], edge_input, face_input,
                self.a0, self.a1, self.coa2, self.b1, self.b2
            )
            preds = self.decoder(x_0_out)  # [sensors, pred_window]
            preds = preds.T  # [pred_window, sensors]
            outputs.append(preds)
        outputs = torch.stack(outputs, dim=0)  # [batch, pred_window, sensors]
        return outputs

def main():
    sensors_json = "./outputs/metr-la/nodes.json"
    edges_json = "./outputs/metr-la/edges.json"
    faces_json = "./outputs/metr-la/faces.json"
    readings_npy = "./outputs/metr-la/sensor_readings_normalized.npy"
    norm_params = np.load("./outputs/metr-la/normalization_params.npz")
    means = norm_params["means"]
    stds = norm_params["stds"]
    columns = norm_params["columns"]  # for safety

    traffic_complex = TrafficComplex(sensors_json, edges_json, faces_json)
    readings = np.load(readings_npy)
    num_sensors, num_timesteps = readings.shape
    input_window = 24
    pred_window = 12

    # Validation: only last 12 steps
    val_steps = 12
    train_dataset = TrafficPredictionDataset(readings, input_window, pred_window, start=0, end=num_timesteps - val_steps + 1)
    val_start = num_timesteps - input_window - pred_window
    val_end = val_start + 1 + input_window + pred_window - 1
    val_dataset = TrafficPredictionDataset(
        readings, input_window, pred_window,
        start=val_start, end=val_end
    )
    print("Validation dataset size:", len(val_dataset))  # Should print 1
    # For small val set, batch size 1 is good
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ... [load features and maps] ...
    with open(sensors_json, "r") as f:
        node_list = json.load(f)
    with open(edges_json, "r") as f:
        edge_list = json.load(f)
    with open(faces_json, "r") as f:
        face_list = json.load(f)

    node_feat_keys = ["mean", "std", "min", "max", "median", "nonzero_frac", "missing_frac"]
    node_features = np.stack([[item[k] for k in node_feat_keys] for item in node_list], axis=0)
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_feat_keys = ["mean_diff", "var_diff", "frac_both_nonzero", "max_lag_corr"]
    edge_map = {}
    edge_features = []
    for idx, item in enumerate(edge_list):
        n1, n2 = sorted([str(x) for x in item["nodes"]])
        edge_map[(n1, n2)] = idx
        edge_features.append([item[k] for k in edge_feat_keys])
    edge_features = np.stack(edge_features, axis=0)
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    face_feat_keys = ["mean_joint", "var_joint", "frac_all_nonzero"]
    face_map = {}
    face_features = []
    for idx, item in enumerate(face_list):
        nodes_sorted = tuple(sorted([str(x) for x in item["nodes"]]))
        face_map[nodes_sorted] = idx
        face_features.append([item[k] for k in face_feat_keys])
    face_features = np.stack(face_features, axis=0)
    face_features = torch.tensor(face_features, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = node_features.to(device)
    edge_features = edge_features.to(device)
    face_features = face_features.to(device)
    model = TrafficCCNN(
        traffic_complex, num_sensors, input_window, pred_window,
        node_feat_dim=len(node_feat_keys),
        edge_feat_dim=len(edge_feat_keys),
        face_feat_dim=len(face_feat_keys)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_rmse = float('inf')
    early_stop_counter = 0
    best_model_path = "traffic_ccnn_best.pt"

    print("Starting training...")
    training_time = time.time()
    epochs = 20
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(
                x,
                node_features,
                edge_features,
                face_features,
                edge_map,
                face_map,
            )
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # Validation
        val_rmse, val_mae, val_mape = evaluate_model(
            model, val_loader, device, node_features, edge_features, face_features, edge_map, face_map, means, stds
        )
        final_epoch_time = time.time() - epoch_start_time
        print(f"Epoch completed in {final_epoch_time:.2f} seconds.")
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | VAl MAPE: {val_mape:.2f}%")

        # LR scheduler step (on train loss)
        o = optimizer.param_groups[0]['lr']
        scheduler.step(train_loss)
        if o != optimizer.param_groups[0]['lr']:
            print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']:.6e}")
        # Early stopping on val RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("New best model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 3:
                print(f"Early stopping at epoch {epoch+1}. Reloading best model.")
                model.load_state_dict(torch.load(best_model_path))
                break
    final_training_time = time.time() - training_time
    print(f"Training completed in {final_training_time:.2f} seconds.")
    # Final evaluation
    final_rmse, final_mae, final_mape = evaluate_model(
        model, val_loader, device, node_features, edge_features, face_features, edge_map, face_map, means, stds
    )
    print(f"Final Evaluation: RMSE={final_rmse:.4f}, MAE={final_mae:.4f}, MAPE={final_mape:.2f}%")
    torch.save(model.state_dict(), "traffic_ccnn_final.pt")
    print("Final model saved as traffic_ccnn_final.pt.")

if __name__ == "__main__":
    main()