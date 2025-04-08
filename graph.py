import networkx as nx

G = nx.MultiGraph()
G.add_edge(1, 2)
G.add_edge(1, 2, key='second')
print(G.edges(keys=False))
