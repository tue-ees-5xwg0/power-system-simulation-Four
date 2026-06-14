import pandas as pd
import pytest
from power_grid_model import ComponentType

import power_system_simulation.tap_optimization as tap_optimization
from power_system_simulation.tap_optimization import optimize_tap_position


def test_tap_optimization_does_not_change_grid(lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles
    original_tap_position = lv_grid[ComponentType.transformer]["tap_pos"][0]

    assert optimize_tap_position(lv_grid, load_p, load_q, "loss") in {-1, 0, 1}
    assert optimize_tap_position(lv_grid, load_p, load_q, "voltage") in {-1, 0, 1}
    assert lv_grid[ComponentType.transformer]["tap_pos"][0] == original_tap_position


def test_tap_optimization_rejects_unknown_criterion(lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles

    with pytest.raises(ValueError, match="Criterion"):
        optimize_tap_position(lv_grid, load_p, load_q, "price")


def test_tap_optimization_selects_lowest_loss(monkeypatch, lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles

    def fake_power_flow(grid, batch_update):
        tap_position = grid[ComponentType.transformer]["tap_pos"][0]
        return {"tap_position": tap_position}

    def fake_aggregation(results, time_index):
        losses = {-1: 20.0, 0: 5.0, 1: 10.0}
        return pd.DataFrame({"energy_loss_kwh": [losses[results["tap_position"]]]})

    monkeypatch.setattr(tap_optimization, "run_batch_power_flow", fake_power_flow)
    monkeypatch.setattr(tap_optimization, "aggregate_line_results", fake_aggregation)

    assert optimize_tap_position(lv_grid, load_p, load_q, "loss") == 0
