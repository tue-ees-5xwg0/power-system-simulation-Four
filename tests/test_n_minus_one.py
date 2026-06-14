import copy

import numpy as np
import pandas as pd
import pytest
from power_grid_model import ComponentType

import power_system_simulation.n_minus_one as n_minus_one
from power_system_simulation.graph_processing import EdgeAlreadyDisabledError, IDNotFoundError
from power_system_simulation.n_minus_one import N_MINUS_1_COLUMNS, calculate_n_minus_1


def test_n_minus_1_result_does_not_change_grid(lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles
    original_from_status = lv_grid[ComponentType.line]["from_status"].copy()
    original_to_status = lv_grid[ComponentType.line]["to_status"].copy()

    result = calculate_n_minus_1(lv_grid, load_p, load_q, 8)

    assert result["alternative_line_id"].tolist() == [24]
    assert result.loc[0, "line_id"] in [4, 5, 8, 11, 24]
    assert result.loc[0, "timestamp"] in load_p.index
    np.testing.assert_array_equal(lv_grid[ComponentType.line]["from_status"], original_from_status)
    np.testing.assert_array_equal(lv_grid[ComponentType.line]["to_status"], original_to_status)


def test_n_minus_1_empty_result_has_columns(lv_grid, lv_profiles):
    grid_without_alternative = copy.deepcopy(lv_grid)
    grid_without_alternative[ComponentType.line] = grid_without_alternative[ComponentType.line][:-1].copy()
    load_p, load_q, _ = lv_profiles

    result = calculate_n_minus_1(grid_without_alternative, load_p, load_q, 4)

    assert result.empty
    assert result.columns.tolist() == N_MINUS_1_COLUMNS


def test_n_minus_1_invalid_or_disconnected_line(lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles

    with pytest.raises(IDNotFoundError):
        calculate_n_minus_1(lv_grid, load_p, load_q, 999)
    with pytest.raises(EdgeAlreadyDisabledError):
        calculate_n_minus_1(lv_grid, load_p, load_q, 24)


def test_n_minus_1_switches_correct_lines(monkeypatch, lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles

    def check_changed_grid(grid, used_load_p, used_load_q):
        lines = grid[ComponentType.line]
        outage = lines[lines["id"] == 8][0]
        alternative = lines[lines["id"] == 24][0]
        assert outage["from_status"] == 0
        assert outage["to_status"] == 0
        assert alternative["to_status"] == 1

        line_table = pd.DataFrame(
            {
                "max_loading_pu": [0.5],
                "max_loading_timestamp": [used_load_p.index[0]],
            },
            index=[24],
        )
        return pd.DataFrame(), line_table

    monkeypatch.setattr(n_minus_one, "run_time_series_power_flow", check_changed_grid)

    result = calculate_n_minus_1(lv_grid, load_p, load_q, 8)
    assert result.loc[0, "alternative_line_id"] == 24
