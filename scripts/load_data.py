import h5py
import numpy as np

def load_h5_data(path):
    with h5py.File(path, 'r') as f:
        return f['df']['block0_values'][:]

def load_adjacency_matrix(path):
    return np.load(path)