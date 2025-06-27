import pandas as pd
import numpy as np
import json
import pickle
from collections import defaultdict, OrderedDict
import h5py
import os

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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def load_node_locations(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # data: {node_id: (road_name, position)}
    # Normalize node ids to string for consistency
    return {str(k): (v[0], float(v[1])) for k, v in data.items()}

def get_road_to_nodes(node_locations, dataset):
    """
    Returns an OrderedDict mapping canonical road names to a list of (node_id, position) tuples.
    For PEMS-BAY, merges and normalizes road names as specified.
    For METR-LA, uses original logic.
    """
    from collections import defaultdict, OrderedDict

    def normalize_road_name(name):
        """Normalize road names for robust matching."""
        name = name.lower().replace("freeway", "").replace("parkway", "").replace("-", " ").replace("/", " ").replace("  ", " ").strip()
        name = name.replace("i ", "i")  # "I 880" -> "i880"
        name = name.replace("ca ", "ca")  # "CA 17" -> "ca17"
        name = name.replace("southbay", "south bay")
        name = name.replace("west valley", "west valley")
        name = name.replace("sinclair", "sinclair")
        return name

    road_to_nodes = defaultdict(list)
    for node_id, (road, pos) in node_locations.items():
        if dataset.lower() == "pems-bay":
            norm_road = normalize_road_name(road)
            road_to_nodes[norm_road].append((node_id, pos))
        else:
            road_to_nodes[road].append((node_id, pos))

    if dataset.lower() == "pems-bay":
        # Merge roads according to user instructions with normalized names
        merged_roads = OrderedDict()
        # Defensive: use get with [] if key is missing, so empty list if not present
        sinclair = road_to_nodes.get("sinclair", [])
        i280 = road_to_nodes.get("i280", [])
        west_valley = road_to_nodes.get("west valley", [])
        ca85 = road_to_nodes.get("ca85", [])
        i880 = road_to_nodes.get("i880", [])
        ca17 = road_to_nodes.get("ca17", [])
        bayshore = road_to_nodes.get("bayshore", [])
        guadalupe = road_to_nodes.get("guadalupe", [])
        south_bay = road_to_nodes.get("south bay", [])

        # Sort all chains by position
        sinclair = sorted(sinclair, key=lambda x: x[1])
        i280 = sorted(i280, key=lambda x: x[1])
        west_valley = sorted(west_valley, key=lambda x: x[1])
        ca85 = sorted(ca85, key=lambda x: x[1])
        i880 = sorted(i880, key=lambda x: x[1])
        ca17 = sorted(ca17, key=lambda x: x[1])
        bayshore = sorted(bayshore, key=lambda x: x[1])
        guadalupe = sorted(guadalupe, key=lambda x: x[1])
        south_bay = sorted(south_bay, key=lambda x: x[1])

        # Final 1-cells (as per instructions)
        merged_roads["Sinclair/I280"] = sinclair + i280
        merged_roads["West Valley/CA85"] = west_valley + ca85
        merged_roads["I880/CA17"] = i880 + ca17
        merged_roads["Bayshore"] = bayshore
        merged_roads["Guadalupe"] = guadalupe
        merged_roads["South Bay"] = south_bay

        # Warn for any unclassified roads (should be empty after normalization)
        expected = {"sinclair", "i280", "west valley", "ca85", "i880", "ca17", "bayshore", "guadalupe", "south bay"}
        for road in road_to_nodes:
            if road not in expected:
                print(f"Unclassified road in PEMS-BAY: '{road}' with nodes {road_to_nodes[road]}")
        return merged_roads
    else:
        # METR-LA or default: sort each road by position
        for road in road_to_nodes:
            road_to_nodes[road].sort(key=lambda x: x[1])
        # Keep as ordered dict for consistent output
        return OrderedDict(sorted(road_to_nodes.items(), key=lambda x: x[0]))

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
    # User option: "metr-la" or "pems-bay"
    dataset = "pems-bay"  # Change as needed

    if dataset.lower() == "metr-la":
        h5_file = "../data/METR-LA/metr-la.h5"
        node_locations_pkl = "../data/METR-LA/node_locations.pkl"
        key = "df"
        output_dir = "./outputs/metr-la/"
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
    elif dataset.lower() == "pems-bay":
        h5_file = "../data/PEMS-BAY/pems-bay.h5"
        node_locations_pkl = "../data/PEMS-BAY/node_locations.pkl"
        key = "speed"
        output_dir = "./outputs/pems-bay/"
        face_specs = [
            (["South Bay", "I880/CA17"],        ["401489", "401464", "413845", "413026"]),
            (["South Bay", "Bayshore"],         ["404554", "401351", "400545", "404522"]),
            (["Bayshore", "Guadalupe"],         ["400760", "401816", "401817", "400911"]),
            (["Bayshore", "I880/CA17"],         ["404759", "400045", "400394", "400922"]),
            (["Bayshore", "Sinclair/I280"],     ["400665", "400400", "401403", "402361"]),
            (["Guadalupe", "I880/CA17"],        ["401440", "403225", "409528", "409524"]),
            (["Guadalupe", "Sinclair/I280"],    ["400236", "403409", "400916", "414284"]),
            (["Guadalupe", "West Valley/CA85"], ["401555", "401495", "400268", "400842"]),
            (["Sinclair/I280", "I880/CA17"],    ["407710", "407711", "401611", "408907"]),
            (["Sinclair/I280", "West Valley/CA85"], ["401846", "407344", "404640", "407370"]),
            (["West Valley/CA85", "I880/CA17"], ["400073", "400715", "400822", "400792"]),
        ]
    else:
        raise ValueError("Unsupported dataset!")

    os.makedirs(output_dir, exist_ok=True)

    # --- Load node locations and .h5 data ---
    node_locations = load_node_locations(node_locations_pkl)
    sensor_ids = list(node_locations.keys())
    df = pd.read_hdf(h5_file, key=key)
    df.columns = df.columns.map(str)

    # Ensure all sensors in node_locations are present in df; add missing columns as NaN
    missing_in_df = [sid for sid in sensor_ids if sid not in df.columns]
    for sid in missing_in_df:
        df[sid] = np.nan
    df = df[sensor_ids]

    sensors = sensor_ids

    # --- Train/val split ---
    val_steps = 12
    train_df = df.iloc[:-val_steps]
    val_df = df.iloc[-val_steps:]

    # --- Normalization: compute on train only
    means = train_df.mean(axis=0)
    stds = train_df.std(axis=0) + 1e-6
    train_df_norm = (train_df - means) / stds
    val_df_norm = (val_df - means) / stds
    train_df_norm_reset = train_df_norm.reset_index(drop=True)
    val_df_norm_reset = val_df_norm.reset_index(drop=True)
    df_norm = pd.concat([train_df_norm_reset, val_df_norm_reset], axis=0, ignore_index=True)

    # --- Save normalization params ---
    np.savez(os.path.join(output_dir, "normalization_params.npz"), means=means.values, stds=stds.values, columns=sensors)

    # --- Node features from normalized train ---
    sensor_stats = compute_sensor_stats(train_df_norm)
    nodes = [{"id": sid, **sensor_stats[sid]} for sid in sensors]
    export_json(nodes, os.path.join(output_dir, "nodes.json"))

    # --- Edges from normalized train ---
    road_to_nodes = get_road_to_nodes(node_locations, dataset)
    edges = make_edges(road_to_nodes, train_df_norm)
    export_json(edges, os.path.join(output_dir, "edges.json"))

    # --- Faces from normalized train ---
    faces = []
    for roads, node_ids in face_specs:
        node_ids_str = [str(n) for n in node_ids]
        face_stats = compute_face_stats(train_df_norm, node_ids_str)
        faces.append({
            "nodes": node_ids_str,
            "roads": roads,
            **face_stats
        })
    export_json(faces, os.path.join(output_dir, "faces.json"))

    # --- Save normalized readings for all data (train+val) ---
    readings_norm = df_norm[sensors].values.T  # [num_sensors, num_timesteps]
    np.save(os.path.join(output_dir, "sensor_readings_normalized.npy"), readings_norm)

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges, {len(faces)} faces/intersections to {output_dir}")

if __name__ == "__main__":
    main()