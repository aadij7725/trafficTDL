import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
from itertools import combinations

# Parameters
DISTANCE_THRESHOLD = 0.2
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 3


def load_sensor_csv(csv_path):
    """Load sensor time-series data from a single CSV file."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df.values  # shape: [T, N]


def compute_sensor_distance_matrix(sensor_data):
    """Compute pairwise distances between sensors based on their time-series behavior."""
    sensor_vectors = sensor_data.T  # shape: [N, T]
    return squareform(pdist(sensor_vectors, metric='euclidean'))


def define_0_cells(n):
    return list(range(n))


def define_1_cells(dist_matrix, threshold):
    return [(i, j) for i in range(len(dist_matrix)) for j in range(i + 1, len(dist_matrix))
            if dist_matrix[i][j] < threshold]


def define_2_cells(dist_matrix, eps, min_samples):
    mds = MDS(dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    clusters = {}
    for i, label in enumerate(db.labels_):
        if label != -1:
            clusters.setdefault(label, []).append(i)
    triangles = [triplet for cluster in clusters.values() if len(cluster) >= 3
                 for triplet in combinations(cluster, 3)]
    return triangles


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


if __name__ == '__main__':
    datasets = {
        "metr-la": "documents/traffictdl/data/metr-la/",
        "pems-bay": "documents/traffictdl/data/pems-bay/pems-bay_cleaned.csv"
    }

    for name, path in datasets.items():
        print(f"Checking path: {path}")
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

        # Compute boundary matrices
        b1 = compute_b1(cells_0, cells_1)
        b2 = compute_b2(cells_1, cells_2)

        # Compute coadjacency matrices
        a01 = compute_adjacency_from_boundary(b1)  # 0-1 adjacency
        a12 = compute_adjacency_from_boundary(b2)  # 1-2 adjacency

        # Save outputs to dataset-specific subfolder
        output_dir = f"./outputs/{name}"
        save_all_matrices(output_dir, b1=b1, b2=b2, a01=a01, a12=a12)

        print(f"{name} matrices saved to {output_dir}")