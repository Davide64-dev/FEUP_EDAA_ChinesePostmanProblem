import csv

import numpy as np

from pulp import PULP_CBC_CMD, LpBinary, LpMinimize, LpProblem, LpVariable, lpSum, LpInteger

from Airline import Airline
from Airport import Airport
from multigraph import MultiGraph
from visualizer import Visualizer
from collections import defaultdict, deque
import networkx as nx


class RouteNetwork:
    def __init__(self):
        self.airports = {}
        self.airlines = {}
        self.graph = MultiGraph()

    @staticmethod
    def from_edge_list(filepath: str) -> "RouteNetwork":
        self = RouteNetwork()
        # the first line of the file is the number of nodes
        # the rest of the lines are edges, separated by spaces
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            n_nodes = int(lines[0].strip())
            self.graph.add_nodes_from(range(n_nodes))
            for line in lines[1:]:
                u, v = map(int, line.strip().split())
                self.graph.add_edge(u, v)
        return self

    def load_airports(self, filepath):
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                airport = Airport(
                    row["Code"],
                    row["Name"],
                    row["City"],
                    row["Country"],
                    row["Latitude"],
                    row["Longitude"],
                )
                self.airports[airport.code] = airport
                self.graph.add_node(
                    airport.code, **vars(airport)
                )  # Store all as node attributes

    def load_airlines(self, filepath):
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                airline = Airline(
                    row["Code"], row["Name"], row["Callsign"], row["Country"]
                )
                self.airlines[airline.code] = airline

    def load_routes(self, filepath):
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row["Source"]
                target = row["Target"]
                airline_code = row["Airline"]

                if source in self.airports and target in self.airports:
                    self.graph.add_edge(source, target, airline=airline_code)

    def print_sample_edges(self, n=5):
        print(f"\nSample of {n} edges:")
        for u, v, key, data in list(self.graph.edges(keys=True, data=True))[:n]:
            print(f"{u} -> {v} via {key}, airline: {data['airline']}")

    def check_if_euler_path_exists(self):
        odd_degree_nodes_count = 0
        degrees = np.array([degree for _, degree in self.graph.degree()])  # type: ignore
        odd_degree_nodes_count = np.sum(degrees % 2 == 1)
        # An Eulerian path exists if there are exactly 0 or 2 vertices of odd degree
        return odd_degree_nodes_count in [0, 2]

    def find_euler_path_hierholzer(self, visualizer: Visualizer | None = None):
        if not self.check_if_euler_path_exists():
            print("No Eulerian path exists in the graph.")
            return None

        stack = []
        circuit = []
        edge_stack = []  # Track used edges for visualization

        # Start at a node with odd degree (if any), otherwise arbitrary
        odd_nodes = [node for node, degree in self.graph.degree() if degree % 2 == 1]
        current_node = odd_nodes[0] if odd_nodes else list(self.graph.nodes())[0]

        if visualizer:
            visualizer.draw_frame(current_node, text=f"Start at node {current_node}")

        while stack or self.graph.degree(current_node) > 0:
            if self.graph.degree(current_node) == 0:
                circuit.append(current_node)
                previous_node = current_node
                current_node = stack.pop()
                u, v, k = edge_stack.pop()

                # Mark the finalized edge green
                if visualizer:
                    visualizer.update_edge_color(u, v, k, "green")
                    visualizer.draw_frame(
                        current_node,
                        text=f"Backtrack from {previous_node} to {current_node} (finalize edge)",
                    )
            else:
                stack.append(current_node)
                next_edge = next(self.graph.edges(current_node, keys=True))  # type: ignore
                u, v, k = next_edge

                # Remove the used edge
                self.graph.remove_edge(u, v, key=k)

                # Track it for later coloring
                edge_stack.append((u, v, k))

                if visualizer:
                    visualizer.update_edge_color(u, v, k, "blue")  # Actively traversed
                    visualizer.draw_frame(
                        v,
                        text=f"Traverse edge {u} → {v} (key={k})",
                    )

                current_node = v

        # Append the last node and finalize the last frame
        circuit.append(current_node)
        if visualizer:
            visualizer.draw_frame(
                text=f"End at {current_node}. Eulerian path complete.",
            )
            visualizer.make_gif()

        circuit.reverse()
        return circuit

    def find_euler_path_fleury(self):
        if not self.check_if_euler_path_exists():
            print("No Eulerian path exists in the graph.")
            return None

        # Find an Eulerian path using Fleury's algorithm
        stack = []
        circuit = []

        odd_nodes = [node for node, degree in self.graph.degree() if degree % 2 == 1]
        current_node = (
            odd_nodes[0] if len(odd_nodes) > 0 else list(self.graph.nodes())[0]
        )

        while stack or self.graph.degree(current_node) > 0:  # type: ignore
            if self.graph.degree(current_node) == 1:
                circuit.append(current_node)
                current_node = stack.pop()
            else:
                stack.append(current_node)
                next_edge = next(self.graph.edges(current_node, keys=True))  # type: ignore
                self.graph.remove_edge(next_edge[0], next_edge[1], key=next_edge[2])
                current_node = next_edge[1]
        circuit.append(current_node)
        circuit.reverse()
        return circuit

    def find_euler_path(self, visualizer: Visualizer | None = None):
        """Implementation agnostic method to find an Eulerian path."""
        return self.find_euler_path_hierholzer(visualizer)

    def csp_find_euler_path(self, visualizer: Visualizer | None = None):
        """
        Attempts to find an Eulerian path using a constraint satisfaction approach.
        If the graph has an Eulerian path, it computes and returns it directly.
        Otherwise, it uses a greedy heuristic that respects degree constraints to approximate a path.
        Visualization highlights traversal and decisions.
        """
        if self.check_if_euler_path_exists():
            print("The graph has an Eulerian path. Finding it...")
            return self.find_euler_path(visualizer)  # Reuse existing visualized version

        print("The graph is not Eulerian. Using CSP-inspired heuristic method...")

        current_node = list(self.graph.nodes())[0]
        visited_edges = set()
        path = [current_node]

        if visualizer:
            visualizer.draw_frame(current_node, text=f"Start at node {current_node}")

        while len(visited_edges) < self.graph.number_of_edges():
            # Only consider edges not yet visited
            neighbors = [
                e
                for e in self.graph.edges(current_node, keys=True)
                if e not in visited_edges
            ]

            if not neighbors:
                if visualizer:
                    visualizer.draw_frame(
                        current_node,
                        text="Dead end reached — no unvisited edges.",
                    )
                break

            # Heuristic: prefer edge to node with lowest remaining degree
            next_edge = min(neighbors, key=lambda e: self.graph.degree(e[1]))  # type: ignore
            u, v, k = next_edge

            visited_edges.add(next_edge)
            path.append(v)

            if visualizer:
                visualizer.update_edge_color(u, v, k, "blue")
                visualizer.draw_frame(
                    v,
                    text=f"Traverse edge {u} → {v} (key={k})",
                )
                visualizer.update_edge_color(u, v, k, "green")

            current_node = v

        if visualizer:
            visualizer.draw_frame(
                current_node,
                text=f"End at node {current_node}. Heuristic path complete.",
            )
            visualizer.make_gif()

        print(path)
        return path


    def simple_approx_eulerian_path(self):
        """
        Simple approximation: 
        - Pair odd-degree nodes arbitrarily.
        - Add edges between each pair.
        - Then find Eulerian path.
        """

        G = self.graph

        # Step 1: Find odd-degree nodes
        odd_nodes = [v for v in G.nodes() if G.degree(v) % 2 == 1]

        # Step 2: Pair nodes in order and add edges
        # If odd number of odd nodes, this will not be a perfect solution
        # but usually graphs have even number of odd nodes
        for i in range(0, len(odd_nodes), 2):
            if i+1 < len(odd_nodes):
                u, v = odd_nodes[i], odd_nodes[i+1]
                # Add an edge between u and v (this is a "virtual" edge if original edge doesn't exist)
                G.add_edge(u, v)

        # Step 3: Now graph should be Eulerian
        # Use NetworkX built-in Eulerian path finder

        temp = RouteNetwork()

        temp.graph = G
        if temp.check_if_euler_path_exists():
            path = list(temp.find_euler_path_hierholzer())
            return path
        else:
            # fallback - no Eulerian path possible (very rare here)
            print("Graph is not Eulerian after simple augmentation.")
            return []

