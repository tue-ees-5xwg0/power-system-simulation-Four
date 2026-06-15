import numpy as np
import pandas as pd
import pytest

from power_system_simulation.LV_grid_analysis import LVGridAnalysis

# --------------------------------------------------
# MOCK POWER FLOW (IMPORTANT)
# --------------------------------------------------

def mock_run_time_series_power_flow(grid, load_p, load_q):
    time_index = load_p.index

    voltage_table = pd.DataFrame({
        "max_u_pu": [1.05] * len(time_index),
        "max_u_pu_node_id": [1] * len(time_index),
        "min_u_pu": [0.95] * len(time_index),
        "min_u_pu_node_id": [2] * len(time_index),
    }, index=time_index)

    line_table = pd.DataFrame({
        "line_id": [1, 2],
        "max_loading_pu": [0.8, 0.9],
        "max_loading_timestamp": [time_index[0], time_index[1]],
        "min_loading_pu": [0.2, 0.3],
        "min_loading_timestamp": [time_index[2], time_index[3]],
        "energy_loss_kwh": [10, 15],
    }).set_index("line_id")

    return voltage_table, line_table


# --------------------------------------------------
# TEST DATA
# --------------------------------------------------

@pytest.fixture
def sample_grid():
    return {
        "nodes": [{"id": i} for i in range(5)],
        "lines": [
            {"id": 1, "from_node": 0, "to_node": 1, "from_status": 1, "to_status": 1},
            {"id": 2, "from_node": 1, "to_node": 2, "from_status": 1, "to_status": 1},
            {"id": 3, "from_node": 1, "to_node": 3, "from_status": 1, "to_status": 1},
            {"id": 4, "from_node": 3, "to_node": 4, "from_status": 0, "to_status": 0},
        ],
        "sources": [{"node": 0}],
        "transformers": [{"id": 10, "tap_position": 0}],
    }


@pytest.fixture
def profiles():
    index = pd.date_range("2024-01-01", periods=5, freq="h")

    load_p = pd.DataFrame({
        2: np.ones(5),
        3: np.ones(5),
        4: np.ones(5),
    }, index=index)

    load_q = load_p.copy()

    ev_profiles = pd.DataFrame({
        "ev1": np.ones(5),
        "ev2": np.ones(5),
        "ev3": np.ones(5),
        "ev4": np.ones(5),
    }, index=index)

    return load_p, load_q, ev_profiles


# --------------------------------------------------
# 1. INPUT VALIDATION
# --------------------------------------------------

def test_invalid_transformer(sample_grid, profiles):
    load_p, load_q, ev = profiles

    sample_grid["transformers"] = []  # invalid

    with pytest.raises(ValueError):
        LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)


def test_invalid_feeder(sample_grid, profiles):
    load_p, load_q, ev = profiles

    with pytest.raises(ValueError):
        LVGridAnalysis(sample_grid, [999], load_p, load_q, ev)


def test_timestamp_mismatch(sample_grid, profiles):
    load_p, load_q, ev = profiles

    load_q = load_q.copy()
    load_q.index = pd.date_range("2025-01-01", periods=5, freq="h")

    with pytest.raises(ValueError):
        LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

def test_invalid_source(sample_grid, profiles):
    load_p, load_q, ev = profiles

    sample_grid["sources"] = []

    with pytest.raises(ValueError):
        LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

def test_load_id_mismatch(sample_grid, profiles):
    load_p, load_q, ev = profiles

    load_q = load_q.rename(columns={2: 999})

    with pytest.raises(ValueError):
        LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)


# --------------------------------------------------
# 2. EV PENETRATION
# --------------------------------------------------

def test_ev_penetration_changes_load(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    original_sum = sim.load_p.sum().sum()

    sim.apply_ev_penetration(0.5, seed=42)

    assert sim.load_p.sum().sum() > original_sum


def test_ev_reproducibility(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    sim1 = LVGridAnalysis(sample_grid, [1], load_p.copy(), load_q, ev)
    sim2 = LVGridAnalysis(sample_grid, [1], load_p.copy(), load_q, ev)

    res1 = sim1.apply_ev_penetration(0.5, seed=42)
    res2 = sim2.apply_ev_penetration(0.5, seed=42)

    assert res1[0].equals(res2[0])

def test_ev_pool_too_small(sample_grid, profiles):
    load_p, load_q, ev = profiles

    ev = ev.iloc[:, :1]  # too few

    with pytest.raises(ValueError):
        LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

def test_full_ev_penetration(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p.copy(), load_q, ev)

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    sim.apply_ev_penetration(1.0)

    # All loads should increase
    assert sim.load_p.sum().sum() > load_p.sum().sum()


# --------------------------------------------------
# 3. TAP OPTIMIZATION
# --------------------------------------------------

def test_optimize_tap(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    best = sim.optimize_tap(range(-2, 3), criterion="loss")

    assert best in range(-2, 3)


def test_invalid_tap_criterion(sample_grid, profiles):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

    with pytest.raises(ValueError):
        sim.optimize_tap(range(-2, 3), criterion="invalid")

def test_optimize_tap_voltage(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    result = sim.optimize_tap(range(-2, 3), criterion="voltage")

    assert result in range(-2, 3)


# --------------------------------------------------
# 4. N-1 ANALYSIS
# --------------------------------------------------

def test_n_minus_1(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    df = sim.n_minus_1(1)

    assert isinstance(df, pd.DataFrame)


def test_n_minus_1_empty(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    # Make all lines enabled → no alternatives
    for line in sample_grid["lines"]:
        line["from_status"] = 1
        line["to_status"] = 1

    sim = LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    df = sim.n_minus_1(1)

    assert df.empty

def test_n_minus_1_invalid_edge(sample_grid, profiles):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p, load_q, ev)

    with pytest.raises(Exception):  # noqa: B017
        sim.n_minus_1(999)


# --------------------------------------------------
# 5. EDGE CASE
# --------------------------------------------------

def test_zero_ev(sample_grid, profiles, monkeypatch):
    load_p, load_q, ev = profiles

    sim = LVGridAnalysis(sample_grid, [1], load_p.copy(), load_q, ev)

    monkeypatch.setattr(
        "power_system_simulation.LV_grid_analysis.run_time_series_power_flow",
        mock_run_time_series_power_flow
    )

    sim.apply_ev_penetration(0.0)

    assert sim.load_p.equals(load_p)
