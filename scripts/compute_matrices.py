import numpy as np
from itertools import combinations

def compute_b1(cells_0, cells_1):
    b1 = np.zeros((len(cells_0), len(cells_1)))
    for j, (u, v) in enumerate(cells_1):
        b1[u, j] = -1
        b1[v, j] = 1
    return b1

def compute_b2(cells_1, cells_2):
    edge_to_index = {tuple(sorted(e)): i for i, e in enumerate(cells_1)}
    b2 = np.zeros((len(cells_1), len(cells_2)))
    for j, triangle in enumerate(cells_2):
        edges = list(combinations(sorted(triangle), 2))
        for edge in edges:
            idx = edge_to_index.get(tuple(sorted(edge)))
            if idx is not None:
                b2[idx, j] = 1
    return b2

def compute_laplacians(b1, b2):
    l1 = b1.T @ b1
    l2 = b2.T @ b2
    return l1, l2