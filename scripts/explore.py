import pickle
import os

def explore_node_locations(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return

    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Print the type of the object
    print(f"Loaded object type: {type(data)}")

    # If it's a dictionary, inspect the contents
    if isinstance(data, dict):
        print(f"Number of entries: {len(data)}")
        print("\nSample entries:")
        for i, (key, value) in enumerate(data.items()):
            print(f"{key}: {value}")
            if i >= 320:  # limit output
                break
    else:
        print("Object is not a dictionary. Showing raw contents:")
        print(data)

if __name__ == "__main__":
    explore_node_locations("../data/PEMS-BAY/node_locations.pkl")