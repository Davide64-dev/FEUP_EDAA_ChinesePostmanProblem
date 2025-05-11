from collections import defaultdict
from typing import Any, Dict, Hashable, Iterable, Optional, Tuple, Union


class MultiGraph:
    def __init__(self) -> None:
        self._adjacency: Dict[Hashable, Dict[Hashable, Dict[int, Dict[str, Any]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        # example of `self._adjacency` structure:
        # {
        #     "a": {
        #         "b": {
        #             0: {"weight": 5},
        #             1: {"weight": 3}
        #         }
        #     },
        #     "b": {
        #         "a": {
        #             0: {"weight": 5},
        #             1: {"weight": 3}
        #         }
        #     }
        # }
        self._node_attributes: Dict[Hashable, Dict[str, Any]] = {}
        self._edge_id_counter: int = 0

    def add_node(self, node: Hashable, **attr: Any) -> None:
        """Add a single node to the graph with optional attributes."""
        if node not in self._node_attributes:
            self._node_attributes[node] = {}
        self._node_attributes[node].update(attr)

    def add_nodes_from(
        self, nodes: Iterable[Union[Hashable, Tuple[Hashable, Dict[str, Any]]]]
    ) -> None:
        """Add multiple nodes to the graph."""
        for n in nodes:
            if isinstance(n, tuple):
                node, attrs = n
                self.add_node(node, **attrs)
            else:
                self.add_node(n)

    def add_edge(
        self, u: Hashable, v: Hashable, key: Optional[int] = None, **attr: Any
    ) -> int:
        """Add an edge between u and v. Returns the edge key."""
        self.add_node(u)
        self.add_node(v)

        if key is None:
            key = self._edge_id_counter
            self._edge_id_counter += 1

        self._adjacency[u][v][key] = attr.copy()
        self._adjacency[v][u][key] = attr.copy()
        return key

    def add_edges_from(
        self,
        edges: Iterable[
            Union[Tuple[Hashable, Hashable], Tuple[Hashable, Hashable, Dict[str, Any]]]
        ],
    ) -> None:
        """Add multiple edges to the graph."""
        for e in edges:
            if len(e) == 2:
                u, v = e
                self.add_edge(u, v)
            elif len(e) == 3:
                u, v, attrs = e
                self.add_edge(u, v, **attrs)

    def remove_node(self, node: Hashable) -> None:
        """Remove a node and all its edges."""
        if node not in self._node_attributes:
            raise KeyError(f"Node {node} not found.")
        for neighbor in list(self._adjacency[node]):
            del self._adjacency[neighbor][node]
        del self._adjacency[node]
        del self._node_attributes[node]

    def remove_edge(self, u: Hashable, v: Hashable, key: Optional[int] = None) -> None:
        """Remove an edge between u and v. If key is None, remove arbitrary edge."""
        if u not in self._adjacency or v not in self._adjacency[u]:
            raise KeyError(f"Edge between {u} and {v} not found.")

        if key is None:
            key = next(iter(self._adjacency[u][v]))

        del self._adjacency[u][v][key]
        del self._adjacency[v][u][key]

        if not self._adjacency[u][v]:
            del self._adjacency[u][v]
        if not self._adjacency[v][u]:
            del self._adjacency[v][u]

    def has_node(self, node: Hashable) -> bool:
        """Check if the node is in the graph."""
        return node in self._node_attributes

    def has_edge(self, u: Hashable, v: Hashable) -> bool:
        """Check if an edge exists between u and v."""
        return (
            u in self._adjacency
            and v in self._adjacency[u]
            and len(self._adjacency[u][v]) > 0
        )

    def nodes(self) -> Iterable[Hashable]:
        """Return an iterable of nodes."""
        return self._node_attributes.keys()

    def edges(
        self,
        nbunch: Optional[Hashable | Iterable[Hashable]] = None,
        data: bool = False,
        keys: bool = False,
    ) -> Iterable:
        """Return an iterable of edges with optional data and keys, limited to nbunch if provided."""
        if nbunch is not None and not isinstance(nbunch, (list, set)):
            nbunch = [nbunch]

        seen = set()
        nodes = set(nbunch) if nbunch is not None else self._adjacency.keys()
        for u in nodes:
            if u not in self._adjacency:
                continue
            for v in self._adjacency[u]:
                if (u, v) in seen or (v, u) in seen:
                    continue
                for k, attr in self._adjacency[u][v].items():
                    if keys and data:
                        yield (u, v, k, attr)
                    elif keys:
                        yield (u, v, k)
                    elif data:
                        yield (u, v, attr)
                    else:
                        yield (u, v)
                seen.add((u, v))

    def neighbors(self, node: Hashable) -> Iterable[Hashable]:
        """Return an iterable of neighbors of a node."""
        if node not in self._adjacency:
            raise KeyError(f"Node {node} not found.")
        return self._adjacency[node].keys()

    def degree(
        self, node: Optional[Hashable] = None
    ) -> Union[int, Iterable[Tuple[Hashable, int]]]:
        """Return the degree of a node, or (node, degree) pairs for all nodes."""
        if node is not None:
            return sum(len(self._adjacency[node][nbr]) for nbr in self._adjacency[node])
        else:
            return (
                (n, sum(len(self._adjacency[n][nbr]) for nbr in self._adjacency[n]))
                for n in self._adjacency
            )

    def number_of_edges(
        self, u: Optional[Hashable] = None, v: Optional[Hashable] = None
    ) -> int:
        """Return the number of edges between u and v, or the total number of edges in graph."""
        assert (u is None and v is None) or (
            u is not None and v is not None
        ), "Either both u and v should be None or both should be provided."

        if u is None and v is None:
            return sum(
                len(self._adjacency[u][v])
                for u in self._adjacency
                for v in self._adjacency[u]
            )
        if u in self._adjacency and v in self._adjacency[u]:
            return len(self._adjacency[u][v])
        return 0

    @property
    def is_multigraph(self) -> bool:
        """Always True for this implementation."""
        return True
