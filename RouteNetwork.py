import csv

import numpy as np
from pulp import PULP_CBC_CMD, LpBinary, LpMinimize, LpProblem, LpVariable, lpSum

from Airline import Airline
from Airport import Airport
from multigraph import MultiGraph


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

    def find_euler_path_hierholzer(self):
        if not self.check_if_euler_path_exists():
            print("No Eulerian path exists in the graph.")
            return None

        # Find an Eulerian path using Hierholzer's algorithm
        stack = []
        circuit = []
        current_node = list(self.graph.nodes())[0]
        while stack or self.graph.degree(current_node) > 0:  # type: ignore
            if self.graph.degree(current_node) == 0:
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

    def find_euler_path_fleury(self):
        if not self.check_if_euler_path_exists():
            print("No Eulerian path exists in the graph.")
            return None

        # Find an Eulerian path using Fleury's algorithm
        stack = []
        circuit = []
        current_node = list(self.graph.nodes())[0]
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

    def find_euler_path(self):
        """Implementation agnostic method to find an Eulerian path."""
        return self.find_euler_path_hierholzer()

    def csp_find_euler_path(self):
        """
        Attempts to find an Eulerian path using a constraint satisfaction approach.
        If the graph has an Eulerian path, it computes and returns it directly.
        Otherwise, it uses a greedy heuristic that respects degree constraints to approximate a path.
        """
        if self.check_if_euler_path_exists():
            print("The graph has an Eulerian path. Finding it...")
            return self.find_euler_path()

        print("The graph is not Eulerian. Using CSP-inspired heuristic method...")

        current_node = list(self.graph.nodes())[0]
        visited_edges = set()
        path = [current_node]

        while len(visited_edges) < self.graph.number_of_edges():
            # Only consider edges not yet visited
            neighbors = [
                e
                for e in self.graph.edges(current_node, keys=True)
                if e not in visited_edges
            ]

            if not neighbors:
                print("Dead end reached. No more unvisited edges.")
                break

            # Heuristic: prefer edges to nodes with lower remaining degree
            next_edge = min(neighbors, key=lambda e: self.graph.degree(e[1]))  # type: ignore

            visited_edges.add(next_edge)
            current_node = next_edge[1]
            path.append(current_node)

        return path

    def ilp_find_euler_path(self):
        """
        Attempts to find an Eulerian path using Integer Linear Programming (ILP).
        If the graph has an Eulerian path, it returns it directly.
        Otherwise, solves an ILP to approximate the minimum-cost traversal covering all edges.
        """
        if self.check_if_euler_path_exists():
            print("The graph has an Eulerian path. Finding it...")
            return self.find_euler_path()

        print(
            "The graph is not Eulerian. Solving ILP to approximate an optimal path..."
        )

        # ILP model
        prob = LpProblem("EulerianPathILP", LpMinimize)

        # Decision variables: x[u,v,k] == 1 if edge (u,v,k) is in the path
        edge_vars = {
            (u, v, k): LpVariable(f"x_{u}_{v}_{k}", cat=LpBinary)
            for u, v, k in self.graph.edges(keys=True)
        }

        # Objective: minimize total number of edges (can be adjusted to minimize weight)
        prob += lpSum(edge_vars[e] for e in edge_vars)

        # Constraints: each edge must be used exactly once
        for u, v, k in self.graph.edges(keys=True):
            prob += edge_vars[(u, v, k)] == 1

        # Optional: degree balance constraints (flow conservation)
        for node in self.graph.nodes():
            in_edges = [e for e in edge_vars if e[1] == node]
            out_edges = [e for e in edge_vars if e[0] == node]
            prob += lpSum(edge_vars[e] for e in in_edges) == lpSum(
                edge_vars[e] for e in out_edges
            )

        # Solve the ILP
        prob.solve(PULP_CBC_CMD(msg=False))

        # Extract the path
        used_edges = [e for e in edge_vars if edge_vars[e].varValue == 1]
        if not used_edges:
            print("ILP solver found no valid solution.")
            return []

        # Reconstruct the path by chaining edges
        from collections import defaultdict

        edge_map = defaultdict(list)
        for u, v, k in used_edges:
            edge_map[u].append((v, k))

        start_node = used_edges[0][0]
        path = [start_node]
        current_node = start_node

        while edge_map[current_node]:
            next_node, _ = edge_map[current_node].pop(0)
            path.append(next_node)
            current_node = next_node

        return path
