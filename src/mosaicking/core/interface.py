from pathlib import Path
from typing import Optional, Sequence, Iterable

import cv2
import networkx as nx
import numpy as np
import numpy.typing as npt
from functools import partial, wraps
import itertools

from mosaicking.core.db import SQLDB, Node


class ImageGraph(nx.DiGraph):

    def _check_db_backend(self):
        if self._db_backend is None:
            raise RuntimeError("Database backend is not initialized, either initialize ImageGraph with a 'db_dir': Path attribute or use ImageGraph.new_db.")

    @staticmethod
    def _validate_db_backend(func):
        """Decorator to ensure _db_backend is not None before running a method."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._check_db_backend()  # Run the check
            return func(self, *args, **kwargs)  # Call the original method
        return wrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._db_backend = None
        if "db_dir" in self.graph:
            self.new_db(kwargs["db_dir"])

    @property
    def db_path(self) -> Path:
        return self._db_backend.db_path

    def new_db(self, db_dir: Path):
        db_dir.mkdir(exist_ok=True, parents=True)
        self._db_backend = SQLDB(db_dir / "mosaicking.db")
        if "db_dir" not in self.graph or db_dir != self.graph["db_dir"]:
            self.graph.update({"db_dir": db_dir})

    @_validate_db_backend
    def register_node(self, name: str, dimensions: tuple[int, int],
                      K: Optional[npt.NDArray[float]] = None,
                      D: Optional[npt.NDArray[float]] = None,
                      overwrite: Optional[bool] = False,
                      **metadata) -> Node:
        node = self._db_backend.register_node(name, dimensions, K, D, overwrite)  # Create the node entry in the db
        self.add_node(node, **metadata)  # register the Node in the graph
        return node

    @_validate_db_backend
    def add_features(self, node: Node, features: dict[str, dict[str, npt.NDArray[np.float32 | np.uint8] | Sequence[cv2.KeyPoint]]]):
        # add the feature data to the db
        self._db_backend.add_node_features(node, features)
        # instead of adding the feature data to the node, add a prepared handle to the db to retrieve the feature data.
        self.nodes[node].update({"features": partial(self._db_backend.get_node_features, node)})

    @_validate_db_backend
    def add_global_features(self, node:Node, global_features: dict[str, npt.NDArray[np.float32]]):
        self._db_backend.add_node_global_features(node, global_features)
        self.nodes[node].update({"global_features": partial(self._db_backend.get_node_global_features, node)})

    @_validate_db_backend
    def register_edge(self, node_from: Node, node_to: Node ):
        self._db_backend.register_edge(node_from, node_to)
        self.add_edge(node_from, node_to)

    @_validate_db_backend
    def add_matches(self, node_from: Node, node_to: Node, matches: dict[str, Sequence[cv2.DMatch]]):
        self._db_backend.add_edge_matches(node_from, node_to, matches)
        self.edges[node_from, node_to].update({"matches": partial(self._db_backend.get_edge_matches, node_from, node_to)})

    @_validate_db_backend
    def add_registration(self, node_from: Node, node_to: Node, registration: npt.NDArray[float], reproj_error: float, frac_inliers: float):
        self._db_backend.add_edge_registration(node_from, node_to, registration, reproj_error, frac_inliers)
        self.edges[node_from, node_to].update({"registration": partial(self._db_backend.get_edge_registration, node_from, node_to)})

    @_validate_db_backend
    def remove_node(self, n: Node):
        super().remove_node(n)
        self._db_backend.remove_node(n)

    @_validate_db_backend
    def remove_edge(self, u: Node, v: Node):
        super().remove_edge(u, v)
        self._db_backend.remove_edge(u, v)

    def remove_nodes_from(self, nodes: Iterable[Node]):
        super().remove_nodes_from(nodes)
        for node in nodes:
            self.remove_node(node)

def _propagate_homographies(graph: ImageGraph) -> npt.NDArray[np.float32]:
    # Given a directed graph of homographies, return them as a sequence of homographies for each node.
    if len(graph) == 1:
        return graph[list(graph.nodes)[0]].get("H0", np.eye(3)[None, ...])
    elif len(graph) < 1:
        raise ValueError("No nodes in graph.")
    else:
        # order the edges by dependency
        sorted_edges = list(nx.topological_sort(nx.line_graph(graph)))
        sorted_H = [graph[u][v]['registration']()[0] for u, v in sorted_edges]
        N0 = sorted_edges[0][0]  # first node in sorted graph
        # if node has an initial rotation
        H0 = graph[N0].get("H0", np.eye(3))
        sorted_H = [H0] + sorted_H
        sorted_H = np.stack(sorted_H, axis=0)
        return np.array(tuple(
            itertools.accumulate(sorted_H, np.matmul)))  # Propagate the homographys to reference to first node.

def _get_graph_image_dimensions(graph: ImageGraph) -> npt.NDArray[np.float32]:
    """Accumulate the 'dimensions' attribute of nodes into a Nx2 NumPy array."""
    if len(graph) < 1:
        raise ValueError("No nodes in graph.")
    return np.array([node.dimensions for node in graph.nodes()])
