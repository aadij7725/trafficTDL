import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations

def load_adjacency_matrix(filepath):
    """Load the adjacency or distance matrix from a .npy file."""
    return np.load(filepath)

def build_spatial_clusters(adj_matrix, eps=0.1, min_samples=3):
    """
    Cluster nodes based on spatial proximity using DBSCAN.
    Nodes are clustered based on distance in the adjacency matrix.
    Returns a list of clusters, each cluster is a list of node indices.
    """
    # Compute a simple distance embedding using MDS-like approach if necessary
    from sklearn.manifold import MDS
    dist_matrix = 1 / (adj_matrix + 1e-5)  # prevent div-by-zero
    dist_matrix[adj_matrix == 0] = 1e6  # mark disconnected as distant

    mds = MDS(dissimilarity='precomputed', n_components=2, random_state=42)
    embedded_coords = mds.fit_transform(dist_matrix)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embedded_coords)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # noise
        clusters.setdefault(label, []).append(idx)

    return list(clusters.values())

def generate_2_cells_from_clusters(clusters):
    """
    Generate candidate 2-cells (triangles) from each cluster.
    """
    two_cells = []
    for cluster in clusters:
        if len(cluster) >= 3:
            for triplet in combinations(cluster, 3):
                two_cells.append(triplet)
    return two_cells

# Example usage:
if __name__ == "__main__":
    adj = load_adjacency_matrix('./data/METR-LA/adj_mx.npy')  # adjust path as needed
    clusters = build_spatial_clusters(adj)
    two_cells = generate_2_cells_from_clusters(clusters)
    print(f"Generated {len(two_cells)} candidate 2-cells.")   
    
    print(two_cells[:5])  # show a few examples 