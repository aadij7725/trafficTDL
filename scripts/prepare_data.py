import pandas as pd
import numpy as np
import json
from itertools import combinations
import h5py

def print_h5_structure(file_name):
    with h5py.File(file_name, 'r') as f:
        def printname(name):
            print(name)
        f.visit(printname)

print_h5_structure("../data/METR-LA/metr-la.h5")

def compute_sensor_stats(df):
    """Compute stats over time for each sensor."""
    stats = {}
    for sensor in df.columns:
        series = df[sensor].astype(float)
        stats[sensor] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "nonzero_frac": float((series != 0).sum() / len(series)),
            "missing_frac": float(series.isna().sum() / len(series)),
        }
    return stats

def infer_edges_by_corr(df, threshold=0.8):
    """Infer edges by correlation threshold between sensors."""
    corr = df.corr().abs()
    edges = set()
    sensors = df.columns
    for i, j in combinations(sensors, 2):
        if corr.loc[i, j] >= threshold:
            edges.add(tuple(sorted((i, j))))
    return list(edges)

def find_triangles(edges):
    """Find all triangles (cliques of 3) in the edge list."""
    # Make adjacency for quick lookup
    from collections import defaultdict
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    # For each triple, check if it's a triangle
    triangles = set()
    for a, b in edges:
        for c in adj[a].intersection(adj[b]):
            triangle = tuple(sorted([a, b, c]))
            triangles.add(triangle)
    return list(triangles)

def export_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

if __name__ == "__main__":
    h5_file = "../data/METR-LA/metr-la.h5"  #
    df = pd.read_hdf(h5_file, key="df")

    # --- DATA EXTRACTION FOR CCNN TRAINING ---
    # Export the sensor readings as a numpy array aligned to sensor order
    sensors = list(df.columns)
    readings = df[sensors].values.T  # shape: [num_sensors, num_timesteps]
    np.save("sensor_readings_aligned.npy", readings)
    print(f"Saved sensor readings aligned as shape {readings.shape} to sensor_readings_aligned.npy")
    # -----------------------------------------

    # 0-cells: nodes
    sensor_stats = compute_sensor_stats(df)
    nodes = [{"id": sid, **sensor_stats[sid]} for sid in sensors]
    export_json(nodes, "nodes.json")

    # 1-cells: edges by correlation threshold (can adjust threshold as needed)
    edges = infer_edges_by_corr(df, threshold=0.8)
    export_json(edges, "edges.json")

    # 2-cells: triangles (3-cliques in edge set)
    triangles = find_triangles(edges)
    export_json(triangles, "triangles.json")

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges, {len(triangles)} triangles.")
