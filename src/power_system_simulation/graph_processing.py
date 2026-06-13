"""Functions for finding downstream vertices and alternative edges."""

import networkx as nx
import pandas as pd


class IDNotFoundError(Exception):
    pass


class InputLengthDoesNotMatchError(Exception):
    pass


class IDNotUniqueError(Exception):
    pass


class GraphNotFullyConnectedError(Exception):
    pass


class GraphCycleError(Exception):
    pass


class EdgeAlreadyDisabledError(Exception):
    pass


class InvalidEdgeTableError(ValueError):
    pass


class GraphProcessor:
    """
    Processes an undirected graph where enabled edges must form a spanning tree
    (fully connected, no cycles). A source vertex is designated as the root.

    Supports:
    - find_downstream_vertices: all vertices on the far side of an edge from the source.
    - find_alternative_edges: disabled edges that can replace a given enabled edge.
    """

    EDGE_COLUMNS = ("edge_id", "from_vertex", "to_vertex", "enabled")

    def __init__(
        self,
        edge_table: pd.DataFrame,
        source_vertex_id: int,
        vertex_ids: list[int] | None = None,
    ) -> None:
        """Initialize an undirected graph from a table with one row per edge."""
        if not isinstance(edge_table, pd.DataFrame):
            raise InvalidEdgeTableError("edge_table must be a pandas DataFrame.")

        missing_columns = set(self.EDGE_COLUMNS) - set(edge_table.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise InvalidEdgeTableError(f"edge_table is missing required columns: {missing}.")

        self.edge_table = edge_table.loc[:, self.EDGE_COLUMNS].copy()
        if self.edge_table.isna().any().any():
            raise InvalidEdgeTableError("edge_table cannot contain missing values.")
        if not self.edge_table.empty and not pd.api.types.is_bool_dtype(self.edge_table["enabled"]):
            raise InvalidEdgeTableError("The enabled column must contain boolean values.")

        edge_ids = self.edge_table["edge_id"].tolist()
        if len(edge_ids) != len(set(edge_ids)):
            raise IDNotUniqueError("Duplicate edge IDs.")

        if vertex_ids is None:
            vertex_ids = list(
                dict.fromkeys(self.edge_table["from_vertex"].tolist() + self.edge_table["to_vertex"].tolist())
            )
        elif len(vertex_ids) != len(set(vertex_ids)):
            raise IDNotUniqueError("Duplicate vertex IDs.")

        vertex_set = set(vertex_ids)
        edge_vertex_id_pairs = list(
            self.edge_table.loc[:, ["from_vertex", "to_vertex"]].itertuples(index=False, name=None)
        )
        for u, v in edge_vertex_id_pairs:
            if u not in vertex_set or v not in vertex_set:
                raise IDNotFoundError("Edge references unknown vertex.")

        if source_vertex_id not in vertex_set:
            raise IDNotFoundError("Invalid source_vertex_id.")

        self.source = source_vertex_id
        self.edge_pairs = dict(zip(edge_ids, edge_vertex_id_pairs, strict=True))
        self.edge_enabled = dict(zip(edge_ids, self.edge_table["enabled"], strict=True))

        # Build graph with only enabled edges
        self._graph = nx.Graph()
        self._graph.add_nodes_from(vertex_ids)
        enabled_edge_count = 0
        for eid, (u, v), en in zip(edge_ids, edge_vertex_id_pairs, self.edge_table["enabled"], strict=True):
            if en:
                enabled_edge_count += 1
                self._graph.add_edge(u, v, edge_id=eid)

        if not nx.is_connected(self._graph):
            raise GraphNotFullyConnectedError("Graph is not fully connected.")
        if enabled_edge_count != len(vertex_ids) - 1:
            raise GraphCycleError("Graph contains a cycle.")

        self._tree = nx.bfs_tree(self._graph, self.source)

    def find_downstream_vertices(self, edge_id: int) -> list[int]:
        """
        Given an edge id, return all the vertices which are in the downstream of the edge,
            with respect to the source vertex.
            Including the downstream vertex of the edge itself!

        Only enabled edges should be taken into account in the analysis.
        If the given edge_id is a disabled edge, it should return empty list.
        If the given edge_id does not exist, it should raise IDNotFoundError.


        For example, given the following graph (all edges enabled):

            vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4

        Call find_downstream_vertices with edge_id=1 will return [2, 4]
        Call find_downstream_vertices with edge_id=3 will return [4]

        Args:
            edge_id: edge id to be searched

        Returns:
            A list of all downstream vertices.
        """
        if edge_id not in self.edge_pairs:
            raise IDNotFoundError(f"Edge {edge_id} not found.")
        if not self.edge_enabled[edge_id]:
            return []

        u, v = self.edge_pairs[edge_id]
        downstream = v if self._tree.has_edge(u, v) else u
        return [downstream] + list(nx.descendants(self._tree, downstream))

    def find_alternative_edges(self, disabled_edge_id: int) -> list[int]:
        """
        Given an enabled edge, do the following analysis:
            If the edge is going to be disabled,
                which (currently disabled) edge can be enabled to ensure
                that the graph is again fully connected and acyclic?
            Return a list of all alternative edges.
        If the disabled_edge_id is not a valid edge id, it should raise IDNotFoundError.
        If the disabled_edge_id is already disabled, it should raise EdgeAlreadyDisabledError.
        If there are no alternative to make the graph fully connected again, it should return empty list.


        For example, given the following graph:

        vertex_0 (source) --edge_1(enabled)-- vertex_2 --edge_9(enabled)-- vertex_10
                 |                               |
                 |                           edge_7(disabled)
                 |                               |
                 -----------edge_3(enabled)-- vertex_4
                 |                               |
                 |                           edge_8(disabled)
                 |                               |
                 -----------edge_5(enabled)-- vertex_6

        Call find_alternative_edges with disabled_edge_id=1 will return [7]
        Call find_alternative_edges with disabled_edge_id=3 will return [7, 8]
        Call find_alternative_edges with disabled_edge_id=5 will return [8]
        Call find_alternative_edges with disabled_edge_id=9 will return []

        Args:
            disabled_edge_id: edge id (which is currently enabled) to be disabled

        Returns:
            A list of alternative edge ids.
        """
        if disabled_edge_id not in self.edge_pairs:
            raise IDNotFoundError(f"Edge {disabled_edge_id} not found.")
        if not self.edge_enabled[disabled_edge_id]:
            raise EdgeAlreadyDisabledError(f"Edge {disabled_edge_id} is already disabled.")

        downstream_vertices = set(self.find_downstream_vertices(disabled_edge_id))

        return [
            edge_id
            for edge_id, (u, v) in self.edge_pairs.items()
            if not self.edge_enabled[edge_id] and ((u in downstream_vertices) != (v in downstream_vertices))
        ]
