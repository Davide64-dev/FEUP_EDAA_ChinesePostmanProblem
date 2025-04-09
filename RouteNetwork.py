import csv
import networkx as nx
from Airline import Airline
from Airport import Airport

class RouteNetwork:
    def __init__(self):
        self.airports = {}     # code -> Airport
        self.airlines = {}     # code -> Airline
        self.graph = nx.MultiGraph()

    def load_airports(self, filepath):
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                airport = Airport(
                    row['Code'], row['Name'], row['City'],
                    row['Country'], row['Latitude'], row['Longitude']
                )
                self.airports[airport.code] = airport
                self.graph.add_node(airport.code, **vars(airport))  # Store all as node attributes

    def load_airlines(self, filepath):
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                airline = Airline(
                    row['Code'], row['Name'], row['Callsign'], row['Country']
                )
                self.airlines[airline.code] = airline

    def load_routes(self, filepath):
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row['Source']
                target = row['Target']
                airline_code = row['Airline']

                if source in self.airports and target in self.airports:
                    self.graph.add_edge(
                        source,
                        target,
                        key=airline_code,
                        airline=airline_code
                    )

    def print_sample_edges(self, n=5):
        print(f"\nSample of {n} edges:")
        for u, v, key, data in list(self.graph.edges(keys=True, data=True))[:n]:
            print(f"{u} -> {v} via {key}, airline: {data['airline']}")

    def check_if_euler_path_exists(self):
        odd_degree_nodes_count = 0
        for node, degree in self.graph.degree():
            if degree % 2 == 1:
                odd_degree_nodes_count += 1
        # An Eulerian path exists if there are exactly 0 or 2 vertices of odd degree
        return odd_degree_nodes_count in [0, 2]
    