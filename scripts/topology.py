import numpy as np
from sklearn.cluster import DBSCAN
import toponetx as tnx

class TrafficComplex:
    def __init__(self, sensors, roads, epsilon=0.01):
        """
        sensors: dict mapping sensor_id to (x, y) coordinates
        roads: list of (sensor_id_1, sensor_id_2) tuples
        epsilon: float, DBSCAN epsilon for 2-cell grouping
        """
        self.sensors = sensors
        self.roads = roads
        self.epsilon = epsilon
        self.complex = tnx.CombinatorialComplex()
        self._build_complex()

    def _build_complex(self):
        # 0-cells: sensors
        for sensor_id in self.sensors:
            self.complex.add_cell([sensor_id], rank=0)

        # 1-cells: roads
        for road in self.roads:
            s1, s2 = road
            self.complex.add_cell([s1, s2], rank=1, name=f"{s1}_{s2}")

        # 2-cells: groups of 3 connected roads found via DBSCAN
        # 1. Create edge midpoints for clustering
        edge_midpoints = []
        edge_to_sensors = []
        for s1, s2 in self.roads:
            x1, y1 = self.sensors[s1]
            x2, y2 = self.sensors[s2]
            midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
            edge_midpoints.append(midpoint)
            edge_to_sensors.append((s1, s2))

        # 2. Cluster midpoints
        if len(edge_midpoints) >= 3:
            X = np.array(edge_midpoints)
            clustering = DBSCAN(eps=self.epsilon, min_samples=3).fit(X)
            labels = clustering.labels_

            # 3. For each cluster, find triplets of connected sensors
            from collections import defaultdict
            cluster_edges = defaultdict(list)
            for idx, label in enumerate(labels):
                if label != -1:
                    cluster_edges[label].append(edge_to_sensors[idx])

            for cluster, edges in cluster_edges.items():
                # Get unique sensors in the cluster
                sensors_in_cluster = set()
                for e in edges:
                    sensors_in_cluster.update(e)
                # Find all triangles (triplets fully connected)
                from itertools import combinations
                for triplet in combinations(sensors_in_cluster, 3):
                    # Check if all three edges exist
                    triplet_edges = [
                        (triplet[0], triplet[1]),
                        (triplet[1], triplet[2]),
                        (triplet[0], triplet[2])
                    ]
                    if all((e in edges or (e[1], e[0]) in edges) for e in triplet_edges):
                        self.complex.add_cell(list(triplet), rank=2, name=f"triangle_{triplet}")

    def get_complex(self):
        return self.complex

# Example usage:
if __name__ == "__main__":
    # Example data (replace with actual sensor/road info)
    sensors = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (0.5, 0.866),
        'D': (2.0, 0.0),
    }
    roads = [
        ('A', 'B'),
        ('B', 'C'),
        ('A', 'C'),
        ('B', 'D')
    ]
    complex_builder = TrafficComplex(sensors, roads, epsilon=1.0)
    print(complex_builder.get_complex())