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
    def __init__(self, readings, input_window, pred_window=12):
        self.readings = readings
        self.input_window = input_window
        self.pred_window = pred_window
        self.num_sensors, self.num_timesteps = readings.shape
        self.starts = np.arange(self.num_timesteps - input_window - pred_window + 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        try:
            start = self.starts[idx]
            x = self.readings[:, start : start + self.input_window]
            y = self.readings[:, start + self.input_window : start + self.input_window + self.pred_window]
            return torch.tensor(x.T, dtype=torch.float32), torch.tensor(y.T, dtype=torch.float32)
        except Exception as e:
            print(f"Data loading error at index {idx}: {e}")
            raise e

def compute_metrics(y_true, y_pred):
    if hasattr(y_true, "detach"): y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"): y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return rmse, mae, mape

def evaluate_model(model, dataloader, device, node_features, edge_features, face_features, edge_map, face_map):
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
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
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
    readings_npy = "./outputs/metr-la/sensor_readings_aligned.npy"

    traffic_complex = TrafficComplex(sensors_json, edges_json, faces_json)

    readings = np.load(readings_npy)
    num_sensors, num_timesteps = readings.shape
    input_window = 24
    pred_window = 12

    # Load node, edge, and face metadata
    with open(sensors_json, "r") as f:
        node_list = json.load(f)
    with open(edges_json, "r") as f:
        edge_list = json.load(f)
    with open(faces_json, "r") as f:
        face_list = json.load(f)

    # Ensure readings and node_list are in the same order (by node id)
    node_ids_json = [item["id"] for item in node_list]
    node_ids_readings = [str(i) for i in node_ids_json]
    # readings.npy should be in the same order as node_list for each axis=0
    # If you generated readings.npy by stacking columns in node_list order, this will be OK.

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

    dataset = TrafficPredictionDataset(readings, input_window, pred_window)
    print("Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=os.cpu_count() // 2)
    print("Dataloader created.")
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
    print("Starting training...")
    training_start_time = time.time()
    epochs = 15
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        losses = []
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
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
            losses.append(loss.item())
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch completed in {epoch_time:.2f} seconds.")
        print(f"Epoch {epoch + 1} | Loss: {np.mean(losses):.6f}")
    training_time = time.time() - training_start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    rmse, mae, mape = evaluate_model(
        model, dataloader, device, node_features, edge_features, face_features, edge_map, face_map
    )
    print(f"Evaluation metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
    torch.save(model.state_dict(), "traffic_ccnn.pt")
    print("Model saved as traffic_ccnn.pt")

if __name__ == "__main__":
    main()