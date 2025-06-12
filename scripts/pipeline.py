from load_data import load_h5_data, load_adjacency_matrix
from build_complex import define_0_cells, define_1_cells, define_2_cells
from compute_matrices import compute_b1, compute_b2, compute_laplacians
from utils import save_matrix

# Load raw data
data = load_h5_data('./data/METR-LA/metr-la.h5')
adj = load_adjacency_matrix('./data/METR-LA/adj_mx.npy')

# Define cells
cells_0 = define_0_cells(data.shape[1])
cells_1 = define_1_cells(adj, threshold=0.1)
cells_2 = define_2_cells(adj)

# Compute boundary and Laplacian matrices
b1 = compute_b1(cells_0, cells_1)
b2 = compute_b2(cells_1, cells_2)
l1, l2 = compute_laplacians(b1, b2)

# Save matrices
save_matrix(b1, './outputs/b1.npy')
save_matrix(b2, './outputs/b2.npy')
save_matrix(l1, './outputs/l1.npy')
save_matrix(l2, './outputs/l2.npy')
