"""
This is a skeleton for the graph processing assignment.

We define a graph processor class with some function skeletons.
"""

import networkx as nx


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


class GraphProcessor:
    """
    Processes an undirected graph where enabled edges must form a spanning tree
    (fully connected, no cycles). A source vertex is designated as the root.

    Supports:
    - find_downstream_vertices: all vertices on the far side of an edge from the source.
    - find_alternative_edges: disabled edges that can replace a given enabled edge.
    """

    def __init__(
        self,
        vertex_ids: list[int],
        edge_ids: list[int],
        edge_vertex_id_pairs: list[tuple[int, int]],
        edge_enabled: list[bool],
        source_vertex_id: int,
    ) -> None:
        """
        Initialize a graph processor object with an undirected graph.
        Only the edges which are enabled are taken into account.
        Check if the input is valid and raise exceptions if not.
        The following conditions should be checked:
            1. vertex_ids and edge_ids should be unique. (IDNotUniqueError)
            2. edge_vertex_id_pairs should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            3. edge_vertex_id_pairs should contain valid vertex ids. (IDNotFoundError)
            4. edge_enabled should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            5. source_vertex_id should be a valid vertex id. (IDNotFoundError)
            6. The graph should be fully connected. (GraphNotFullyConnectedError)
            7. The graph should not contain cycles. (GraphCycleError)
        If one certain condition is not satisfied, the error in the parentheses should be raised.

        Args:
            vertex_ids: list of vertex ids
            edge_ids: liest of edge ids
            edge_vertex_id_pairs: list of tuples of two integer
                Each tuple is a vertex id pair of the edge.
            edge_enabled: list of bools indicating of an edge is enabled or not
            source_vertex_id: vertex id of the source in the graph
        """
        if len(vertex_ids) != len(set(vertex_ids)):
            raise IDNotUniqueError("Duplicate vertex IDs.")
        if len(edge_ids) != len(set(edge_ids)):
            raise IDNotUniqueError("Duplicate edge IDs.")

        if len(edge_vertex_id_pairs) != len(edge_ids):
            raise InputLengthDoesNotMatchError("edge_vertex_id_pairs length mismatch.")
        if len(edge_enabled) != len(edge_ids):
            raise InputLengthDoesNotMatchError("edge_enabled length mismatch.")

        vertex_set = set(vertex_ids)
        for u, v in edge_vertex_id_pairs:
            if u not in vertex_set or v not in vertex_set:
                raise IDNotFoundError("Edge references unknown vertex.")

        if source_vertex_id not in vertex_set:
            raise IDNotFoundError("Invalid source_vertex_id.")

        self.source = source_vertex_id
        self.edge_pairs = dict(zip(edge_ids, edge_vertex_id_pairs, strict=False))
        self.edge_enabled = dict(zip(edge_ids, edge_enabled, strict=False))

        # Build graph with only enabled edges
        self._graph = nx.Graph()
        self._graph.add_nodes_from(vertex_ids)
        for eid, (u, v), en in zip(edge_ids, edge_vertex_id_pairs, edge_enabled, strict=False):
            if en:
                self._graph.add_edge(u, v, edge_id=eid)

        if not nx.is_connected(self._graph):
            raise GraphNotFullyConnectedError("Graph is not fully connected.")
        if not nx.is_tree(self._graph):
            raise GraphCycleError("Graph contains a cycle.")

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
        tree = nx.bfs_tree(self._graph, self.source)

        # The downstream vertex is the one whose parent is the other
        downstream = v if tree.has_edge(u, v) else u
        return [downstream] + list(nx.descendants(tree, downstream))

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

        u, v = self.edge_pairs[disabled_edge_id]
        temp = self._graph.copy()
        temp.remove_edge(u, v)

        comp_source = nx.node_connected_component(temp, self.source)
        comp_other = set(temp.nodes) - comp_source

        return [
            eid
            for eid, (a, b) in self.edge_pairs.items()
            if not self.edge_enabled[eid]
            and ((a in comp_source and b in comp_other) or (a in comp_other and b in comp_source))
        ]
