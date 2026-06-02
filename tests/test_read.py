import pytest
from power_grid_model._core.errors import ConflictID, PowerGridBatchError

from power_system_simulation.pgm_processing import (
    ComponentType,
    aggregate_line_results,
    aggregate_node_voltage_results,
    create_load_batch_update,
    create_pgm,
    read_load_profile,
    read_pgm_json,
    run_batch_power_flow,
    validate_load_profile,
)


def test_read_pgm_json():
    result = read_pgm_json("data/input_network_data.json")

    assert result is not None


def test_create_pgm_invalid_input_raises_error():
    input_data = read_pgm_json("data/input_network_data.json")
    invalid_input_data = input_data.copy()
    invalid_input_data[ComponentType.node] = invalid_input_data[ComponentType.node].copy()
    invalid_input_data[ComponentType.node]["id"][0] = invalid_input_data[ComponentType.node]["id"][1]

    with pytest.raises(ConflictID):
        create_pgm(invalid_input_data)


def test_create_load_batch_update():
    active_profile = read_load_profile("data/active_power_profile.parquet")
    reactive_profile = read_load_profile("data/reactive_power_profile.parquet")
    validate_load_profile(active_profile, reactive_profile)

    batch_update = create_load_batch_update(active_profile, reactive_profile)
    assert batch_update[ComponentType.sym_load]["id"].shape == active_profile.shape
    assert batch_update[ComponentType.sym_load]["id"][0].tolist() == active_profile.columns.tolist()
    ##assert batch_update[ComponentType.sym_load]["id"].tolist() == active_profile.columns.tolist()
    assert batch_update[ComponentType.sym_load]["p_specified"].shape == active_profile.shape
    assert batch_update[ComponentType.sym_load]["q_specified"].shape == reactive_profile.shape


def test_run_batch_power_flow():
    input_data = read_pgm_json("data/input_network_data.json")
    model = create_pgm(input_data)

    active_profile = read_load_profile("data/active_power_profile.parquet")
    reactive_profile = read_load_profile("data/reactive_power_profile.parquet")

    batch_update = create_load_batch_update(active_profile, reactive_profile)

    results = run_batch_power_flow(model, batch_update)

    line_summary = aggregate_line_results(
        results,
        active_profile.index,
    )

    summary = aggregate_node_voltage_results(
        results,
        active_profile.index,
    )

    assert line_summary.shape[0] == len(results[ComponentType.line][0])
    assert "energy_loss_kwh" in line_summary.columns
    assert "max_loading_pu" in line_summary.columns
    assert "min_loading_pu" in line_summary.columns
    assert summary.shape[0] == len(active_profile)
    assert results is not None


def test_run_batch_power_flow_invalid_update_rises_error():
    input_data = read_pgm_json("data/input_network_data.json")
    model = create_pgm(input_data)

    active_profile = read_load_profile("data/active_power_profile.parquet")
    reactive_profile = read_load_profile("data/reactive_power_profile.parquet")

    batch_update = create_load_batch_update(active_profile, reactive_profile)
    batch_update[ComponentType.sym_load]["id"][0][0] = 999

    with pytest.raises(PowerGridBatchError):
        run_batch_power_flow(model, batch_update)


def test_validate_load_profile_mismatching_timestamps():
    active_profile = read_load_profile("data/active_power_profile.parquet")
    reactive_profile = read_load_profile("data/reactive_power_profile.parquet")

    reactive_profile = reactive_profile.copy()
    reactive_profile.index = reactive_profile.index.shift(1, freq="h")

    with pytest.raises(ValueError):
        validate_load_profile(active_profile, reactive_profile)


def test_validate_load_profile_mismatching_load_ids():
    active_profile = read_load_profile("data/active_power_profile.parquet")
    reactive_profile = read_load_profile("data/reactive_power_profile.parquet")

    reactive_profile = reactive_profile.copy()
    reactive_profile.columns = [100, 101, 102]

    with pytest.raises(ValueError):
        validate_load_profile(active_profile, reactive_profile)
