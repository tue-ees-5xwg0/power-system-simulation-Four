import pytest

from power_system_simulation.graph_processing import (
    EdgeAlreadyDisabledError,
    GraphCycleError,
    GraphNotFullyConnectedError,
    GraphProcessor,
    IDNotFoundError,
    IDNotUniqueError,
    InputLengthDoesNotMatchError,
)

# --- shared test graphs ---


@pytest.fixture
def linear_graph():
    # vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4
    return GraphProcessor(
        vertex_ids=[0, 2, 4],
        edge_ids=[1, 3],
        edge_vertex_id_pairs=[(0, 2), (2, 4)],
        edge_enabled=[True, True],
        source_vertex_id=0,
    )


@pytest.fixture
def branched_graph():
    # vertex_0 --edge_1-- vertex_2 --edge_9-- vertex_10
    #    |                   | edge_7(dis)
    #    |--edge_3-- vertex_4 | edge_8(dis)
    #    |--edge_5-- vertex_6
    return GraphProcessor(
        vertex_ids=[0, 2, 4, 6, 10],
        edge_ids=[1, 3, 5, 7, 8, 9],
        edge_vertex_id_pairs=[(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)],
        edge_enabled=[True, True, True, False, False, True],
        source_vertex_id=0,
    )


# --- __init__ validation ---


def test_duplicate_vertex_ids():
    with pytest.raises(IDNotUniqueError):
        GraphProcessor([0, 0], [1], [(0, 0)], [True], 0)


def test_duplicate_edge_ids():
    with pytest.raises(IDNotUniqueError):
        GraphProcessor([0, 1], [1, 1], [(0, 1), (0, 1)], [True, True], 0)


def test_edge_pairs_wrong_length():
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor([0, 1], [1, 2], [(0, 1)], [True, True], 0)


def test_edge_enabled_wrong_length():
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor([0, 1], [1], [(0, 1)], [True, True], 0)


def test_invalid_vertex_in_edge():
    with pytest.raises(IDNotFoundError):
        GraphProcessor([0, 1], [1], [(0, 99)], [True], 0)


def test_invalid_source():
    with pytest.raises(IDNotFoundError):
        GraphProcessor([0, 1], [1], [(0, 1)], [True], 99)


def test_not_connected():
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor([0, 1, 2], [1], [(0, 1)], [True], 0)


def test_cycle():
    with pytest.raises(GraphCycleError):
        GraphProcessor([0, 1, 2], [1, 2, 3], [(0, 1), (1, 2), (0, 2)], [True, True, True], 0)


# --- find_downstream_vertices ---


def test_downstream_nonexistent_edge(linear_graph):
    with pytest.raises(IDNotFoundError):
        linear_graph.find_downstream_vertices(99)


def test_downstream_disabled_edge():
    gp = GraphProcessor(
        vertex_ids=[0, 1, 2],
        edge_ids=[1, 2, 3],
        edge_vertex_id_pairs=[(0, 1), (1, 2), (0, 2)],
        edge_enabled=[True, True, False],
        source_vertex_id=0,
    )
    assert gp.find_downstream_vertices(3) == []


def test_downstream_edge_closest_to_source(linear_graph):
    assert sorted(linear_graph.find_downstream_vertices(1)) == [2, 4]


def test_downstream_edge_furthest_from_source(linear_graph):
    assert sorted(linear_graph.find_downstream_vertices(3)) == [4]


def test_downstream_leaf_edge(branched_graph):
    assert sorted(branched_graph.find_downstream_vertices(9)) == [10]


def test_downstream_branch_edge(branched_graph):
    assert sorted(branched_graph.find_downstream_vertices(1)) == [2, 10]


# --- find_alternative_edges ---


def test_alternative_nonexistent_edge(branched_graph):
    with pytest.raises(IDNotFoundError):
        branched_graph.find_alternative_edges(99)


def test_alternative_already_disabled(branched_graph):
    with pytest.raises(EdgeAlreadyDisabledError):
        branched_graph.find_alternative_edges(7)


def test_alternative_no_options(branched_graph):
    assert branched_graph.find_alternative_edges(9) == []


def test_alternative_one_option(branched_graph):
    assert sorted(branched_graph.find_alternative_edges(1)) == [7]


def test_alternative_two_options(branched_graph):
    assert sorted(branched_graph.find_alternative_edges(3)) == [7, 8]


def test_alternative_other_branch(branched_graph):
    assert sorted(branched_graph.find_alternative_edges(5)) == [8]
