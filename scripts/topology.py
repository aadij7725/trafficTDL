"""
Traffic Topology - Combinatorial Complex representation for traffic sensor network
with 0-cells as sensors, 1-cells as roads (edges), and 2-cells as triangles (groups of 3 connected sensors)
"""

import json
import toponetx as tnx

class TrafficComplex:
    def __init__(self, sensors_path, roads_path, faces_path):
        """
        Initializes the combinatorial complex for the traffic sensor network.
        Args:
          sensors_path: path to JSON file containing list of sensor IDs (0-cells)
          roads_path: path to JSON file containing list of 2-element lists (sensor ids) for each road (1-cells)
          triangles_path: path to JSON file containing list of 3-element lists (sensor ids) for each triangle (2-cells)
        """
        self.sensors = self._load_json(sensors_path)
        self.roads = self._load_json(roads_path)
        self.faces = self._load_json(faces_path)
        
        self.complex = tnx.CombinatorialComplex()
        self._build_complex()
    
    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def _build_complex(self):
        print("Building Traffic Combinatorial Complex...")
        # 0-cells: sensors
        print(f"Adding {len(self.sensors)} sensors as rank 0 cells")
        for id in self.sensors:
            self.complex.add_cell(id["id"], rank=0)
        
        # 1-cells: roads
        print(f"Adding {len(self.roads)} roads as rank 1 cells")
        for road in self.roads:
            sensor_a, sensor_b = road["nodes"]
            name = f"{sensor_a}_{sensor_b}"
            self.complex.add_cell([sensor_a, sensor_b], rank=1, name=name)
        
        # 2-cells: triangles
        print(f"Adding {len(self.faces)} faces as rank 2 cells")
        for triangle in self.faces:
            sensor_a, sensor_b, sensor_c, sensor_d = triangle["nodes"]
            face_name = f"{sensor_a}_{sensor_b}_{sensor_c}{sensor_d}"
            self.complex.add_cell([sensor_a, sensor_b, sensor_c, sensor_d], rank=2, name=face_name)
        
        print("Traffic combinatorial complex built.")
    
    def get_complex(self):
        """Return the constructed combinatorial complex."""
        return self.complex

# Example usage
def main():
    sensors_path = "./outputs/metr-la/nodes.json"     # Path to your 0-cells JSON
    roads_path = "./outputs/metr-la/edges.json"        # Path to your 1-cells JSON
    faces_path = "./outputs/metr-la/faces.json" # Path to your 2-cells JSON

    traffic_complex = TrafficComplex(sensors_path, roads_path, faces_path)
    print("\nCombinatorial complex summary:")
    print(traffic_complex.get_complex())

if __name__ == "__main__":
    main()
