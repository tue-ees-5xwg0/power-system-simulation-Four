import copy

import pandas as pd
import pytest
from power_grid_model import ComponentType

import power_system_simulation.ev_penetration as ev_penetration
from power_system_simulation.ev_penetration import apply_ev_penetration


def test_ev_penetration_is_reproducible_and_does_not_change_load(lv_grid, lv_profiles):
    load_p, load_q, ev_profiles = lv_profiles
    original_load = load_p.copy()

    first_result = apply_ev_penetration(lv_grid, [4, 5], load_p, load_q, ev_profiles, 1.0, seed=10)
    second_result = apply_ev_penetration(lv_grid, [4, 5], load_p, load_q, ev_profiles, 1.0, seed=10)

    assert first_result[0].equals(second_result[0])
    assert load_p.equals(original_load)


def test_invalid_ev_penetration(lv_grid, lv_profiles):
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_ev_penetration(lv_grid, [4, 5], *lv_profiles, penetration=1.5)


def test_ev_penetration_requires_enough_houses_per_feeder(lv_grid, lv_profiles):
    uneven_grid = copy.deepcopy(lv_grid)
    uneven_grid[ComponentType.sym_load]["node"] = [9, 9]

    with pytest.raises(ValueError, match="enough houses"):
        apply_ev_penetration(uneven_grid, [4, 5], *lv_profiles, penetration=1.0)


def test_ev_profiles_are_added_once(monkeypatch, lv_grid, lv_profiles):
    load_p, load_q, ev_profiles = lv_profiles
    used_load_p = None

    def capture_power_flow(grid, updated_load_p, updated_load_q):
        nonlocal used_load_p
        used_load_p = updated_load_p
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(ev_penetration, "run_time_series_power_flow", capture_power_flow)
    apply_ev_penetration(lv_grid, [4, 5], load_p, load_q, ev_profiles, 1.0, seed=10)

    total_added_power = (used_load_p - load_p).sum(axis=1)
    assert total_added_power.equals(ev_profiles.sum(axis=1))
