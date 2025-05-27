import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from RouteNetwork import RouteNetwork

warnings.filterwarnings("ignore")

figures = []
plot_titles = []

for n_nodes in range(5, 21, 5):
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

        # draw graphs for visualization
        # network.graph.draw(show=False)
        # plt.title(f"Graph: {edge_list}, Optimal Path Length: {optimal_path_lenght}")
        # plt.show()

        # circuit = network.find_euler_path_hierholzer()

        # TODO: uncomment the following lines when the greedy approximation is correctly implemented
        # approx = network.greedy_approx_eulerian_path()
        # approx_path_length = len(approx) - 1

        # TODO: and delete this
        approx = []
        approx_path_length = optimal_path_lenght + random.randint(0, 10)

        # print(circuit)
        print(f"Optimal Path Length: {optimal_path_lenght}")
        print(f"Approx Path Length: {approx_path_length}")
        print("Approx Path:")
        print(approx, end="\n\n")

        optimal_path_lengths.append(optimal_path_lenght)
        approx_path_lengths.append(approx_path_length)

    # Prepare data for seaborn
    data = pd.DataFrame(
        {
            "Graph Index": np.arange(len(optimal_path_lengths)).tolist() * 2,
            "Path Length": optimal_path_lengths + approx_path_lengths,
            "Type": ["Optimal"] * len(optimal_path_lengths)
            + ["Approx"] * len(approx_path_lengths),
        }
    )

    figures.append(data)
    plot_titles.append(f"{n_nodes} Nodes")

# Plot all figures in a 2x2 grid at the end
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (data, title) in enumerate(zip(figures, plot_titles)):
    sns.barplot(
        data=data,
        x="Graph Index",
        y="Path Length",
        hue="Type",
        palette="Set2",
        ax=axes[idx],
    )
    axes[idx].set_title(f"Optimal vs Approx Path Length ({title})")
    axes[idx].set_xlabel("Graph Index")
    axes[idx].set_ylabel("Path Length")
    axes[idx].grid(axis="y", linestyle="--", alpha=0.7)
    axes[idx].legend(title="")

# Hide any unused subplots
for j in range(len(figures), 4):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Plot average difference between optimal and approx path lengths vs number of nodes
avg_diffs = []
node_counts = []

for idx, data in enumerate(figures):
    optimal = data[data["Type"] == "Optimal"]["Path Length"].values
    approx = data[data["Type"] == "Approx"]["Path Length"].values
    avg_diff = np.mean(approx - optimal)
    avg_diffs.append(avg_diff)
    node_counts.append(int(plot_titles[idx].split()[0]))

plt.figure(figsize=(8, 6))
sns.lineplot(x=node_counts, y=avg_diffs, marker="o")
plt.title("Average Difference: Approx vs Optimal Path Length by Number of Nodes")
plt.xlabel("Number of Nodes")
plt.ylabel("Average Difference (Approx - Optimal)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
