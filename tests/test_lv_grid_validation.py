import copy

import pytest
from power_grid_model import ComponentType

from power_system_simulation.graph_processing import GraphCycleError, GraphNotFullyConnectedError
from power_system_simulation.lv_grid_validation import (
    create_line_graph,
    validate_ev_profiles,
    validate_feeders,
    validate_grid,
    validate_load_profiles,
)
from power_system_simulation.pgm_processing import InputDataValidationError, ProfileTimestampMismatchError


def test_valid_grid_and_graph(lv_grid):
    graph = validate_grid(lv_grid)

    assert graph.find_downstream_vertices(4) == [6, 9]
    assert graph.find_alternative_edges(8) == [24]
    assert create_line_graph(lv_grid).find_alternative_edges(8) == [24]


@pytest.mark.parametrize("component", [ComponentType.transformer, ComponentType.source])
def test_requires_one_transformer_and_source(lv_grid, component):
    invalid_grid = copy.deepcopy(lv_grid)
    invalid_grid[component] = invalid_grid[component][:0]

    with pytest.raises(ValueError):
        validate_grid(invalid_grid)


@pytest.mark.parametrize("component", [ComponentType.line, ComponentType.sym_load])
def test_requires_lines_and_loads(lv_grid, component):
    invalid_grid = copy.deepcopy(lv_grid)
    del invalid_grid[component]

    with pytest.raises(ValueError):
        validate_grid(invalid_grid)


def test_grid_must_be_valid_pgm_data(lv_grid):
    invalid_grid = copy.deepcopy(lv_grid)
    invalid_grid[ComponentType.node]["id"][0] = invalid_grid[ComponentType.node]["id"][1]

    with pytest.raises(InputDataValidationError):
        validate_grid(invalid_grid)


def test_grid_must_be_connected_and_without_cycles(lv_grid):
    disconnected_grid = copy.deepcopy(lv_grid)
    disconnected_grid[ComponentType.line]["to_status"][2] = 0
    with pytest.raises(GraphNotFullyConnectedError):
        validate_grid(disconnected_grid)

    cyclic_grid = copy.deepcopy(lv_grid)
    cyclic_grid[ComponentType.line]["to_status"][-1] = 1
    with pytest.raises(GraphCycleError):
        validate_grid(cyclic_grid)


def test_feeder_validation(lv_grid):
    validate_feeders(lv_grid, [4, 5])

    with pytest.raises(ValueError, match="At least one"):
        validate_feeders(lv_grid, [])
    with pytest.raises(ValueError, match="valid line IDs"):
        validate_feeders(lv_grid, [999])
    with pytest.raises(ValueError, match="transformer"):
        validate_feeders(lv_grid, [8])


def test_load_profile_validation(lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles
    validate_load_profiles(lv_grid, load_p, load_q)

    with pytest.raises(ProfileTimestampMismatchError):
        validate_load_profiles(lv_grid, load_p, load_q.shift(freq="h"))

    wrong_load_p = load_p.rename(columns={10: 999})
    wrong_load_q = load_q.rename(columns={10: 999})
    with pytest.raises(ValueError, match="sym_load"):
        validate_load_profiles(lv_grid, wrong_load_p, wrong_load_q)


def test_ev_profile_validation(lv_grid, lv_profiles):
    load_p, _, ev_profiles = lv_profiles
    validate_ev_profiles(lv_grid, load_p, ev_profiles)

    with pytest.raises(ValueError, match="timestamps"):
        validate_ev_profiles(lv_grid, load_p, ev_profiles.shift(freq="h"))
    with pytest.raises(ValueError, match="at least one"):
        validate_ev_profiles(lv_grid, load_p, ev_profiles.iloc[:, :1])
