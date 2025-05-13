import os
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import networkx as nx


class Visualizer:
    def __init__(self, graph, output_dir="frames", layout_seed=42):
        self.graph = graph
        self.output_dir = output_dir  # this is for the frames
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)

        # Use a MultiGraph with full edge keys for layout and drawing
        self._nx_graph = nx.MultiGraph()
        self._nx_graph.add_nodes_from(graph.nodes())
        self._nx_graph.add_edges_from(graph.edges(keys=True))
        self.layout = nx.spring_layout(self._nx_graph, seed=layout_seed)

        # Initialize all edge colors to gray
        self.edge_colors = {(u, v, k): "gray" for u, v, k in graph.edges(keys=True)}

    def update_edge_color(self, u, v, key, color):
        edge = self._canonical_edge(u, v, key)
        if edge in self.edge_colors:
            self.edge_colors[edge] = color

    def draw_frame(self, current_node: Optional[str] = None, text: Optional[str] = ""):
        plt.figure(figsize=(8, 6))

        # Draw nodes and labels
        nx.draw_networkx_nodes(
            self._nx_graph,
            self.layout,
            node_color="white",
            edgecolors="black",
            node_size=800,
        )
        nx.draw_networkx_labels(self._nx_graph, self.layout, font_size=10)

        # Draw all multiedges with curvature to separate them visually
        for idx, (u, v, k) in enumerate(self._nx_graph.edges(keys=True)):
            edge = self._canonical_edge(u, v, k)
            color = self.edge_colors.get(edge, "gray")

            # Alternate curvature directions and scale by key index
            curvature = 0.1 if k == 0 else -0.1 if k == 1 else 0.2 * ((-1) ** k)

            nx.draw_networkx_edges(
                self._nx_graph,
                self.layout,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={curvature}",
                edge_color=color,
                width=2,
            )

        # Highlight the current node if given
        if current_node is not None:
            nx.draw_networkx_nodes(
                self._nx_graph,
                pos=self.layout,
                nodelist=[current_node],
                node_color="yellow",
                node_size=900,
            )

        plt.title(text)  # type: ignore
        filename = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
        plt.savefig(filename)
        plt.close()
        self.frame_count += 1

    def make_gif(self, output_file="euler_path.gif", fps=1):
        images = []
        for i in range(self.frame_count):
            path = os.path.join(self.output_dir, f"frame_{i:04d}.png")
            images.append(imageio.imread(path))
        imageio.mimsave(output_file, images, fps=fps)

        # Clean up
        for i in range(self.frame_count):
            os.remove(os.path.join(self.output_dir, f"frame_{i:04d}.png"))
        if not os.listdir(self.output_dir):
            os.rmdir(self.output_dir)

    def _canonical_edge(self, u, v, k):
        return (u, v, k) if (u, v, k) in self.edge_colors else (v, u, k)
