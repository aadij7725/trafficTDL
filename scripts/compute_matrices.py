import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
from itertools import combinations
import torch

# Parameters
DISTANCE_THRESHOLD = 0.05  # Lowered to allow more connections
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 3


def load_sensor_csv(csv_path):
    """Load sensor time-series data from a single CSV file."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df.values  # shape: [T, N]


def compute_sensor_distance_matrix(sensor_data):
    """Compute pairwise distances between sensors based on their time-series behavior.
    Scales the distance matrix to avoid extreme values and clamps large distances."""
    sensor_vectors = sensor_data.T  # shape: [N, T]
    dist_matrix = squareform(pdist(sensor_vectors, metric='euclidean'))
    # Clamp extreme values to a reasonable upper bound (e.g., 99th percentile or fixed value)
    finite_vals = dist_matrix[np.isfinite(dist_matrix)]
    if finite_vals.size > 0:
        upper_bound = np.percentile(finite_vals, 99)
        dist_matrix = np.clip(dist_matrix, 0, upper_bound)
        # Normalize to [0, 1]
        max_dist = np.nanmax(dist_matrix[dist_matrix != np.inf])
        if max_dist > 0:
            dist_matrix /= max_dist
    return dist_matrix


def define_0_cells(n):
    return list(range(n))


def define_1_cells(dist_matrix, threshold, fallback_k=5):
    # Print some statistics about the distance matrix to help debug
    finite_dists = dist_matrix[np.isfinite(dist_matrix)]
    print(f"Min distance: {np.min(finite_dists):.4f}, Max distance: {np.max(finite_dists):.4f}, "
          f"Mean distance: {np.mean(finite_dists):.4f}, Median distance: {np.median(finite_dists):.4f}")
    print(f"Using DISTANCE_THRESHOLD = {threshold}")
    edges = [(i, j) for i in range(len(dist_matrix)) for j in range(i + 1, len(dist_matrix))
             if np.isfinite(dist_matrix[i][j]) and dist_matrix[i][j] < threshold]
    if len(edges) == 0:
        print("No 1-cells found using threshold. Falling back to k-nearest neighbors.")
        edges = set()
        for i in range(len(dist_matrix)):
            neighbors = np.argsort(dist_matrix[i])[:fallback_k + 1]  # include self
            for j in neighbors:
                if i < j:
                    edges.add((i, j))
                elif j < i:
                    edges.add((j, i))
        edges = list(edges)
    return edges


def define_2_cells(dist_matrix, eps, min_samples, max_triangles=10000):
    mds = MDS(dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    clusters = {}
    for i, label in enumerate(db.labels_):
        if label != -1:
            clusters.setdefault(label, []).append(i)

    triangles = set()
    for cluster in clusters.values():
        if len(cluster) >= 3:
            for triplet in combinations(sorted(cluster), 3):
                triangles.add(tuple(sorted(triplet)))
                if len(triangles) >= max_triangles:
                    break
        if len(triangles) >= max_triangles:
            break

    return list(triangles)


def compute_b1(cells_0, cells_1):
    b1 = np.zeros((len(cells_0), len(cells_1)))
    for j, (u, v) in enumerate(cells_1):
        b1[u, j] = -1
        b1[v, j] = 1
    return b1


def compute_b2(cells_1, cells_2):
    edge_to_idx = {tuple(sorted(edge)): i for i, edge in enumerate(cells_1)}
    b2 = np.zeros((len(cells_1), len(cells_2)))
    for j, triangle in enumerate(cells_2):
        for edge in combinations(sorted(triangle), 2):
            idx = edge_to_idx.get(tuple(sorted(edge)))
            if idx is not None:
                b2[idx, j] = 1
    return b2


def compute_adjacency_from_boundary(B):
    return (B @ B.T) != 0


def save_all_matrices(output_folder, **matrices):
    os.makedirs(output_folder, exist_ok=True)
    for name, matrix in matrices.items():
        np.save(os.path.join(output_folder, f"{name}.npy"), matrix)


def get_neighborhood_matrices(cells_0, cells_1, cells_2):
    print("Computing neighborhood matrices...")

    b1_np = compute_b1(cells_0, cells_1)
    b2_np = compute_b2(cells_1, cells_2) if len(cells_2) > 0 else np.zeros((len(cells_1), 0))

    # Adjacency matrices
    a0_np = compute_adjacency_from_boundary(b1_np)
    a1_np = compute_adjacency_from_boundary(b2_np) if len(cells_2) > 0 else np.zeros((len(cells_1), len(cells_1)), dtype=bool)

    # Coadjacency matrix (triangle coadjacency via shared nodes)
    if len(cells_2) > 0:
        b02_np = np.zeros((len(cells_0), len(cells_2)))
        for j, tri in enumerate(cells_2):
            for node in tri:
                b02_np[node, j] = 1
        coa2_np = b02_np.T @ b02_np
        np.fill_diagonal(coa2_np, 0)
    else:
        coa2_np = np.zeros((0, 0))

    # Convert to torch sparse tensors
    to_sparse = lambda x: torch.from_numpy(x).to_sparse()

    return (
        to_sparse(a0_np.astype(np.float32)),
        to_sparse(a1_np.astype(np.float32)),
        to_sparse(coa2_np.astype(np.float32)),
        to_sparse(b1_np.astype(np.float32)),
        to_sparse(b2_np.astype(np.float32))
    )


if __name__ == '__main__':
    datasets = {
        "metr-la": "../data/metr-la/metr-la_cleaned.csv",
        "pems-bay": "../data/pems-bay/pems-bay_cleaned.csv"
    }

    for name, path in datasets.items():
        print(f"Checking path: {path}")
        print(f"Resolved absolute path: {os.path.abspath(path)}")
        if not os.path.exists(path):
            print(f"Skipping {name}: file not found at {path}")
            continue
        print(f"Processing {name} dataset...")

        # Load sensor data
        sensor_data = load_sensor_csv(path)

        # Compute distance matrix
        dist_matrix = compute_sensor_distance_matrix(sensor_data)

        # Define cells
        cells_0 = define_0_cells(sensor_data.shape[1])
        cells_1 = define_1_cells(dist_matrix, DISTANCE_THRESHOLD)
        cells_2 = define_2_cells(dist_matrix, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)

        print(f"Number of 0-cells: {len(cells_0)}")
        print(f"Number of 1-cells: {len(cells_1)}")
        print(f"Number of 2-cells: {len(cells_2)}")

        a0, a1, coa2, b1_tensor, b2_tensor = get_neighborhood_matrices(cells_0, cells_1, cells_2)

        print("a0 (0-cell adjacency):", a0.shape)
        print("a1 (1-cell adjacency):", a1.shape)
        print("coa2 (2-cell coadjacency):", coa2.shape)
        print("b1 (incidence 0->1):", b1_tensor.shape)
        print("b2 (incidence 1->2):", b2_tensor.shape)

        # Save outputs to dataset-specific subfolder
        output_dir = f"./outputs/{name}"
        save_all_matrices(output_dir, b1=b1_tensor.to_dense().numpy(), b2=b2_tensor.to_dense().numpy(),
                          a01=a0.to_dense().numpy(), a12=a1.to_dense().numpy())

        print(f"{name} matrices saved to {output_dir}")
