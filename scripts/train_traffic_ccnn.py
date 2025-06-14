import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from topology import TrafficComplex
from topomodelx.nn.combinatorial.hmc import HMC

# --- DATASET ---
class TrafficPredictionDataset(Dataset):
    def __init__(self, readings, input_window, pred_window=12):
        """
        readings: [num_sensors, num_timesteps]
        input_window: how many timesteps of history to use
        pred_window: always 12 (to match your requirement)
        """
        self.readings = readings
        self.input_window = input_window
        self.pred_window = pred_window
        self.num_sensors, self.num_timesteps = readings.shape
        self.samples = []

        # Build samples: for each possible window, get [input_window] as input, next [pred_window] as target
        for start in range(self.num_timesteps - input_window - pred_window + 1):
            x = readings[:, start : start + input_window]        # [num_sensors, input_window]
            y = readings[:, start + input_window : start + input_window + pred_window]  # [num_sensors, pred_window]
            self.samples.append((x.T, y.T))  # transpose to [input_window, num_sensors], [pred_window, num_sensors]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    

# --- MODEL ---
class TrafficCCNN(nn.Module):

    def scipy_to_torch_sparse(mat):
        mat = mat.tocoo()
        indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
        values = torch.from_numpy(mat.data)
        shape = mat.shape
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).float()

    def __init__(self, complex, num_sensors, input_window, pred_window=12, hidden_dim=64):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_window = input_window
        self.pred_window = pred_window
        self.hidden_dim = hidden_dim

        # HMC layer for topology-aware message passing
        # We'll use a simple structure: 1 layer, from 0-cells to 2-cells
        self.hmc = HMC([
            [
                [hidden_dim, hidden_dim, hidden_dim],  # 0-cells: in, hidden, out
                [hidden_dim, hidden_dim, hidden_dim],  # 1-cells: in, hidden, out
                [hidden_dim, hidden_dim, hidden_dim],  # 2-cells: in, hidden, out
            ]
        ])

        # Temporal encoder (LSTM over input window for each node)
        self.temporal_encoder = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, batch_first=True
        )

        # Final decoder: simple MLP to produce output window for each sensor
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_window)
        )

        # Save topology matrices as buffers (convert to dense for simplicity)
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

    def forward(self, x):
        """
        x: [batch, input_window, num_sensors]
        Predicts: [batch, pred_window, num_sensors]
        """
        batch, win, sensors = x.shape
        assert sensors == self.num_sensors

        # For each node, run LSTM over the input window
        x0 = x.permute(0,2,1).unsqueeze(-1)  # [batch, num_sensors, input_window, 1]
        x0 = x0.reshape(batch * sensors, win, 1)
        _, (h, _) = self.temporal_encoder(x0)  # h: [1, batch * sensors, hidden]
        h = h.squeeze(0).reshape(batch, sensors, self.hidden_dim)  # [batch, num_sensors, hidden_dim]

        # HMC expects [num_0_cells, in_channels]
        # We'll message-pass for each sample in the batch
        outputs = []
        for i in range(batch):
            # Prepare inputs: only 0-cells have features, others are zeros
            x_0 = h[i]             # [num_sensors, hidden_dim]
            x_1 = torch.zeros(self.b1.shape[1], self.hidden_dim, device=x.device)
            x_2 = torch.zeros(self.b2.shape[1], self.hidden_dim, device=x.device)

            # Message passing
            x_0_out, _, _ = self.hmc(
                x_0, x_1, x_2,
                self.a0, self.a1, self.coa2, self.b1, self.b2
            )  # [num_sensors, hidden_dim]

            # Decoder: for each node, produce pred_window output
            preds = self.decoder(x_0_out)  # [num_sensors, pred_window]
            preds = preds.T  # [pred_window, num_sensors]
            outputs.append(preds)
        outputs = torch.stack(outputs, dim=0)  # [batch, pred_window, num_sensors]
        return outputs
    # --- Metrics computation and evaluation ---

def compute_metrics(y_true, y_pred):
    """
    y_true: numpy array or torch tensor of true values
    y_pred: numpy array or torch tensor of predicted values
    Returns: tuple (rmse, mae, mape)
    """
    # Convert to numpy if torch tensor
    if hasattr(y_true, "detach"): y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"): y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true + 1e-8))) * 100
    return rmse, mae, mape

# --- Evaluation after training ---

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return compute_metrics(y_true, y_pred)

# --- TRAINING SCRIPT ---
def main():
    # Paths (edit as needed)
    sensors_json = "./outputs/metr-la/nodes.json"
    roads_json = "./outputs/metr-la/edges.json"
    triangles_json = "./outputs/metr-la/triangles.json"
    readings_npy = "./outputs/metr-la/sensor_readings_aligned.npy"

    # --- Load Topology ---
    traffic_complex = TrafficComplex(
        sensors_json, roads_json, triangles_json
    )

    # --- Load Data ---
    readings = np.load(readings_npy)  # [num_sensors, num_timesteps]
    num_sensors, num_timesteps = readings.shape
    input_window = num_timesteps - 12  # use all but last 12 as input for each sample
    pred_window = 12

    # --- Dataset and DataLoader ---
    dataset = TrafficPredictionDataset(readings, input_window, pred_window)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficCCNN(traffic_complex, num_sensors, input_window, pred_window).to(device)

    # --- Training Setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # --- Training Loop ---
    epochs = 100
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in dataloader:
            x = x.to(device)  # [batch, input_window, num_sensors]
            y = y.to(device)  # [batch, pred_window, num_sensors]
            pred = model(x)   # [batch, pred_window, num_sensors]
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.6f}")

    # --- Save Model ---
    torch.save(model.state_dict(), "traffic_ccnn.pt")
    print("Model saved as traffic_ccnn.pt")

    # --- Evaluate and print metrics ---
    print("Evaluating model on full dataset for metrics...")
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    rmse, mae, mape = evaluate_model(model, eval_loader, device)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()
