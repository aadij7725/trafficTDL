import h5py
import numpy as np
import pandas as pd
import os

def load_h5_data(path, base_dir):
    full_path = os.path.join(base_dir, path)
    try:
        df = pd.read_hdf(full_path)
    except Exception:
        with h5py.File(full_path, 'r') as f:
            data = f['df']['block0_values'][:]
            columns = []
            for col in f['df']['axis0'][:]:
                if isinstance(col, bytes):
                    columns.append(col.decode())
                else:
                    columns.append(str(col))
            df = pd.DataFrame(data, columns=columns)
    return df

def clean_and_export_h5_to_csv(h5_path, csv_path, base_dir):
    df = load_h5_data(h5_path, base_dir)
    # Convert all columns to numeric, coerce errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df.to_csv(csv_path, index=False)
    print(f"Exported cleaned data to {csv_path}")

def load_adjacency_matrix(path):
    return np.load(path)

# Example usage:
clean_and_export_h5_to_csv(
    'metr-la.h5',
    'metr-la_cleaned.csv',
    base_dir='/Users/aadijain/Documents/trafficTDL/data/METR-LA/'
)
clean_and_export_h5_to_csv(
    'pems-bay.h5',
    'pems-bay_cleaned.csv',
    base_dir='/Users/aadijain/Documents/trafficTDL/data/PEMS-BAY/'
)