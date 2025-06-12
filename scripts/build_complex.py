from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
from itertools import combinations
import numpy as np

def define_0_cells(num_nodes):
    return list(range(num_nodes))

def define_1_cells(adj_matrix, threshold=0.1):
    return [(i, j) for i in range(len(adj_matrix)) for j in range(i+1, len(adj_matrix))
            if adj_matrix[i][j] < threshold]

def define_2_cells(adj_matrix, eps=0.1, min_samples=3):
    from sklearn.manifold import MDS
    dist = 1 / (adj_matrix + 1e-5)
    dist[adj_matrix == 0] = 1e6
    mds = MDS(dissimilarity='precomputed')
    coords = mds.fit_transform(dist)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:
            clusters.setdefault(label, []).append(i)
    triangles = [triplet for cluster in clusters.values() if len(cluster) >= 3
                 for triplet in combinations(cluster, 3)]
    return triangles
