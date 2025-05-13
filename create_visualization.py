from RouteNetwork import RouteNetwork
from visualizer import Visualizer

filepath = "./dataset/edge_list_4.txt"
network = RouteNetwork.from_edge_list(filepath)
visualizer = Visualizer(network.graph)

network.find_euler_path_hierholzer(visualizer)
