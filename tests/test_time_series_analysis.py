from power_grid_model import ComponentType

from power_system_simulation.time_series_analysis import run_time_series_power_flow


def test_run_time_series_power_flow(lv_grid, lv_profiles):
    load_p, load_q, _ = lv_profiles

    voltage_table, line_table = run_time_series_power_flow(lv_grid, load_p, load_q)

    assert voltage_table.index.equals(load_p.index)
    assert set(line_table.index) == set(lv_grid[ComponentType.line]["id"])
