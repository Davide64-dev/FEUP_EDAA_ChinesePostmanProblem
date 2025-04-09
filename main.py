from RouteNetwork import RouteNetwork
from Airline import Airline
from Airport import Airport

def main():
    network = RouteNetwork()
    network.load_airports('dataset/airports.csv')
    network.load_airlines('dataset/airlines.csv')
    network.load_routes('dataset/flights.csv')
    network.print_sample_edges()
    print(network.check_if_euler_path_exists())


if __name__ == "__main__":
    main()