import pandas as pd
import numpy as np
import json
import pickle
from collections import defaultdict
import h5py

def compute_sensor_stats(df):
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

def compute_edge_stats(df, node1, node2):
    s1 = df[node1].astype(float)
    s2 = df[node2].astype(float)
    diff = s1 - s2
    both_nonzero = (s1 != 0) & (s2 != 0)
    if both_nonzero.any():
        lag_corr = np.corrcoef(s1[both_nonzero], s2[both_nonzero])[0,1]
    else:
        lag_corr = 0.0
    return {
        "mean_diff": float(diff.mean()),
        "var_diff": float(diff.var()),
        "frac_both_nonzero": float(both_nonzero.sum() / len(s1)),
        "max_lag_corr": float(lag_corr if not np.isnan(lag_corr) else 0.0)
    }

def compute_face_stats(df, node_ids):
    readings = np.stack([df[str(n)].astype(float).values for n in node_ids])
    mean_joint = float(np.mean(readings))
    var_joint = float(np.var(readings))
    frac_all_nonzero = float(np.mean(np.all(readings != 0, axis=0)))
    return {
        "mean_joint": mean_joint,
        "var_joint": var_joint,
        "frac_all_nonzero": frac_all_nonzero
    }

def export_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def load_node_locations(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # data: {node_id: (road_name, position)}
    # Normalize node ids to string for consistency
    return {str(k): (v[0], float(v[1])) for k, v in data.items()}

def get_road_to_nodes(node_locations, merge_santa_ana_golden_state=True):
    road_to_nodes = defaultdict(list)
    for node_id, (road, pos) in node_locations.items():
        # Merge Santa Ana and Golden State
        if merge_santa_ana_golden_state and (road.lower() in ["santa ana freeway", "golden state freeway"]):
            road = "Santa Ana/Golden State"
        road_to_nodes[road].append((node_id, pos))
    for road in road_to_nodes:
        road_to_nodes[road].sort(key=lambda x: x[1])
    return road_to_nodes

def make_edges(road_to_nodes, df):
    edges = []
    for road, nodes in road_to_nodes.items():
        if len(nodes) < 2:
            continue
        for i in range(len(nodes) - 1):
            n1, _ = nodes[i]
            n2, _ = nodes[i+1]
            edge_stats = compute_edge_stats(df, n1, n2)
            edge = {
                "nodes": [n1, n2],
                "road": road,
                **edge_stats
            }
            edges.append(edge)
    return edges

def main():
    h5_file = "../data/METR-LA/metr-la.h5"
    node_locations_pkl = "../data/METR-LA/node_locations.pkl"

    # Load time series data
    df = pd.read_hdf(h5_file, key="df")
    sensors = [str(s) for s in df.columns]

    # Load node locations mapping
    node_locations = load_node_locations(node_locations_pkl)

    # Sanity check: all sensors should be in node_locations
    missing = [s for s in sensors if s not in node_locations]
    assert not missing, f"Missing sensors in node_locations.pkl: {missing}"

    # --- 0-cells: nodes ---
    sensor_stats = compute_sensor_stats(df)
    nodes = [{"id": sid, **sensor_stats[sid]} for sid in sensors]
    export_json(nodes, "nodes.json")

    # --- 1-cells: edges (road chains, merging Santa Ana and Golden State) ---
    road_to_nodes = get_road_to_nodes(node_locations, merge_santa_ana_golden_state=True)
    edges = make_edges(road_to_nodes, df)
    export_json(edges, "edges.json")

    # --- 2-cells: faces (intersections, hardcoded sensors) ---
    # Format: (roads, [sensor ids])
    face_specs = [
        (["San Diego Freeway", "Ventura Freeway"],                [765099, 764781, 764794, 765171]),
        (["Ventura Freeway", "Hollywood Freeway"],                [764858, 718141, 768066, 773939]),
        (["Santa Ana/Golden State", "Ventura Freeway"],           [761604, 774067, 773916, 773927]),
        (["Glendale Freeway", "Ventura Freeway"],                 [718072, 767454, 767455, 717592]),
        (["Glendale Freeway", "Santa Ana/Golden State"],          [716956, 767509, 767495, 767494]),
        (["Arroyo Seco Parkway", "Hollywood Freeway"],            [764853, 718045, 773023, 772513]),
        (["Santa Ana/Golden State", "Arroyo Seco Parkway"],       [773012, 771673, 716949, 771667]),
        (["Glendale Freeway", "Arroyo Seco Parkway"],             [716943, 716941, 716942, 716939]),
        (["Foothill Freeway", "Glendale Freeway"],                [769847, 767610, 767609, 769941])
    ]

    faces = []
    for roads, node_ids in face_specs:
        node_ids_str = [str(n) for n in node_ids]
        face_stats = compute_face_stats(df, node_ids_str)
        faces.append({
            "nodes": node_ids_str,
            "roads": roads,
            **face_stats
        })
    export_json(faces, "faces.json")

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges, {len(faces)} faces/intersections.")

if __name__ == "__main__":
    main()