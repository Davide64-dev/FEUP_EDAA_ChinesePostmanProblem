from RouteNetwork import RouteNetwork


def main():
    network = RouteNetwork()
    network.load_airports("dataset/airports.csv")
    network.load_airlines("dataset/airlines.csv")
    network.load_routes("dataset/flights.csv")
    network.print_sample_edges()

    path_exists = network.check_if_euler_path_exists()
    print(f"\nEuler path exists: {path_exists}")

    if path_exists:
        path: list = network.find_euler_path()  # type: ignore
        print("\nEuler path found:")
        for u, v, key in path:
            print(f"{u} -> {v} via {key}")

    else:
        print("\nNo Euler path found.")


if __name__ == "__main__":
    main()
