import os

import matplotlib.pyplot as plt
import numpy as np

from RouteNetwork import RouteNetwork

n_nodes = 10

edge_lists = [
    f for f in os.listdir("./dataset/evaluation") if f.startswith(str(n_nodes))
]
edge_lists.sort()
edge_lists = sorted(edge_lists, key=lambda x: len(x))
print(edge_lists)

optimal_path_lengths = []
approx_path_lengths = []

for edge_list in edge_lists:
    network = RouteNetwork.from_edge_list(f"./dataset/evaluation/{edge_list}")

    optimal_path_lenght = int(edge_list.split(".")[0].split("_")[2])

    network.graph.draw(show=False)
    plt.title(f"Graph: {edge_list}, Optimal Path Length: {optimal_path_lenght}")
    plt.show()

    # circuit = network.find_euler_path_hierholzer()
    approx = network.simple_approx_eulerian_path()
    approx_path_length = len(approx) - 1

    # print(circuit)
    print(approx)
    print(approx_path_length)

    optimal_path_lengths.append(optimal_path_lenght)
    approx_path_lengths.append(approx_path_length)


# fig = plt.figure(figsize=(10, 6))
# plt.plot(optimal_path_lengths, label="Optimal Path Length", marker="o")
# plt.plot(approx_path_lengths, label="Approx Path Length", marker="o")
# plt.xlabel("Graph Index")
# plt.ylabel("Path Length")
# plt.title("Optimal vs Approx Path Length")
# plt.legend()
# plt.grid()
# plt.show()

fig = plt.figure(figsize=(10, 6))
indices = np.arange(len(optimal_path_lengths))
bar_width = 0.35

plt.bar(
    indices - bar_width / 2,
    optimal_path_lengths,
    width=bar_width,
    label="Optimal Path Length",
    alpha=0.7,
)
plt.bar(
    indices + bar_width / 2,
    approx_path_lengths,
    width=bar_width,
    label="Approx Path Length",
    alpha=0.7,
)
plt.xlabel("Graph Index")
plt.ylabel("Path Length")
plt.title("Optimal vs Approx Path Length")
plt.xticks(indices)
plt.legend()
plt.grid(axis="y")
plt.show()
