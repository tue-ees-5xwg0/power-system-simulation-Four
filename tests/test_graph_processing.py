import pandas as pd
import pytest

from power_system_simulation.graph_processing import (
    EdgeAlreadyDisabledError,
    GraphCycleError,
    GraphNotFullyConnectedError,
    GraphProcessor,
    IDNotFoundError,
    IDNotUniqueError,
    InvalidEdgeTableError,
)

# --- shared test graphs ---


def edge_table(edge_ids, edge_pairs, enabled):
    return pd.DataFrame(
        {
            "edge_id": edge_ids,
            "from_vertex": [pair[0] for pair in edge_pairs],
            "to_vertex": [pair[1] for pair in edge_pairs],
            "enabled": enabled,
        }
    )


@pytest.fixture
def linear_graph():
    # vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4
    return GraphProcessor(edge_table([1, 3], [(0, 2), (2, 4)], [True, True]), source_vertex_id=0)


@pytest.fixture
def branched_graph():
    # vertex_0 --edge_1-- vertex_2 --edge_9-- vertex_10
    #    |                   | edge_7(dis)
    #    |--edge_3-- vertex_4 | edge_8(dis)
    #    |--edge_5-- vertex_6
    return GraphProcessor(
        edge_table(
            [1, 3, 5, 7, 8, 9],
            [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)],
            [True, True, True, False, False, True],
        ),
        source_vertex_id=0,
    )


# --- __init__ validation ---


def test_duplicate_vertex_ids():
    with pytest.raises(IDNotUniqueError):
        GraphProcessor(edge_table([1], [(0, 1)], [True]), 0, vertex_ids=[0, 0, 1])


def test_duplicate_edge_ids():
    with pytest.raises(IDNotUniqueError):
        GraphProcessor(edge_table([1, 1], [(0, 1), (0, 1)], [True, True]), 0)


def test_invalid_vertex_in_edge():
    with pytest.raises(IDNotFoundError):
        GraphProcessor(edge_table([1], [(0, 99)], [True]), 0, vertex_ids=[0, 1])


def test_invalid_source():
    with pytest.raises(IDNotFoundError):
        GraphProcessor(edge_table([1], [(0, 1)], [True]), 99)


def test_not_connected():
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(edge_table([1], [(0, 1)], [True]), 0, vertex_ids=[0, 1, 2])


def test_cycle():
    with pytest.raises(GraphCycleError):
        GraphProcessor(edge_table([1, 2, 3], [(0, 1), (1, 2), (0, 2)], [True, True, True]), 0)


def test_parallel_enabled_edges_are_a_cycle():
    with pytest.raises(GraphCycleError):
        GraphProcessor(edge_table([1, 2], [(0, 1), (0, 1)], [True, True]), 0)


def test_graph_with_one_vertex():
    graph = GraphProcessor(pd.DataFrame(columns=GraphProcessor.EDGE_COLUMNS), 0, vertex_ids=[0])

    assert graph.source == 0


def test_create_from_edge_table():
    edge_table = pd.DataFrame(
        {
            "edge_id": [1, 2, 3],
            "from_vertex": [0, 1, 0],
            "to_vertex": [1, 2, 2],
            "enabled": [True, True, False],
        }
    )

    graph = GraphProcessor(edge_table, source_vertex_id=0)

    assert graph.edge_table.equals(edge_table)
    assert graph.find_downstream_vertices(1) == [1, 2]
    assert graph.find_alternative_edges(1) == [3]


def test_edge_table_can_include_isolated_vertices_for_validation():
    edge_table = pd.DataFrame(
        {
            "edge_id": [1],
            "from_vertex": [0],
            "to_vertex": [1],
            "enabled": [True],
        }
    )

    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(edge_table, source_vertex_id=0, vertex_ids=[0, 1, 2])


@pytest.mark.parametrize(
    "edge_table",
    [
        [{"edge_id": 1}],
        pd.DataFrame({"edge_id": [1]}),
        pd.DataFrame(
            {
                "edge_id": [1],
                "from_vertex": [0],
                "to_vertex": [1],
                "enabled": [None],
            }
        ),
        pd.DataFrame(
            {
                "edge_id": [1],
                "from_vertex": [0],
                "to_vertex": [1],
                "enabled": [1],
            }
        ),
    ],
)
def test_invalid_edge_table(edge_table):
    with pytest.raises(InvalidEdgeTableError):
        GraphProcessor(edge_table, source_vertex_id=0)


# --- find_downstream_vertices ---


def test_downstream_nonexistent_edge(linear_graph):
    with pytest.raises(IDNotFoundError):
        linear_graph.find_downstream_vertices(99)


def test_downstream_disabled_edge():
    gp = GraphProcessor(edge_table([1, 2, 3], [(0, 1), (1, 2), (0, 2)], [True, True, False]), 0)
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
