import numpy as np
import pytest
from power_grid_model import ComponentType
from power_grid_model.errors import PowerGridSerializationError
from power_grid_model.validation import validate_batch_data, validate_input_data

from power_system_simulation.pgm_processing import (
    BatchDataValidationError,
    InputDataValidationError,
    ProfileLoadIDMismatchError,
    ProfileTimestampMismatchError,
    TimeIndexLengthError,
    aggregate_line_results,
    aggregate_node_voltage_results,
    create_load_batch_update,
    create_pgm,
    deserialize_pgm_json,
    read_json_file,
    read_load_profile,
    run_batch_power_flow,
    validate_load_profile,
)


@pytest.fixture
def input_data():
    return deserialize_pgm_json(read_json_file("data/input_network_data.json"))


@pytest.fixture
def load_profiles():
    return (
        read_load_profile("data/active_power_profile.parquet"),
        read_load_profile("data/reactive_power_profile.parquet"),
    )


def test_read_json_file():
    json_data = read_json_file("data/input_network_data.json")

    assert isinstance(json_data, str)
    assert '"version": "1.0"' in json_data


def test_deserialize_pgm_json(input_data):
    assert validate_input_data(input_data) is None
    assert input_data[ComponentType.node]["id"].tolist() == [1, 2, 3, 4]
    assert input_data[ComponentType.line]["id"].tolist() == [5, 6, 7]


def test_deserialize_invalid_json():
    with pytest.raises(PowerGridSerializationError):
        deserialize_pgm_json('{"not": "pgm data"}')


def test_create_pgm(input_data):
    assert create_pgm(input_data) is not None


def test_create_pgm_invalid_input_raises_error(input_data):
    invalid_input_data = input_data.copy()
    invalid_input_data[ComponentType.node] = invalid_input_data[ComponentType.node].copy()
    invalid_input_data[ComponentType.node]["id"][0] = invalid_input_data[ComponentType.node]["id"][1]

    with pytest.raises(InputDataValidationError, match="input data"):
        create_pgm(invalid_input_data)


def test_validate_load_profile(load_profiles):
    active_profile, reactive_profile = load_profiles

    validate_load_profile(active_profile, reactive_profile)


def test_validate_load_profile_mismatching_timestamps(load_profiles):
    active_profile, reactive_profile = load_profiles
    reactive_profile = reactive_profile.copy()
    reactive_profile.index = reactive_profile.index.shift(1, freq="h")

    with pytest.raises(ProfileTimestampMismatchError):
        validate_load_profile(active_profile, reactive_profile)


def test_validate_load_profile_mismatching_load_ids(load_profiles):
    active_profile, reactive_profile = load_profiles
    reactive_profile = reactive_profile.copy()
    reactive_profile.columns = [100, 101, 102]

    with pytest.raises(ProfileLoadIDMismatchError):
        validate_load_profile(active_profile, reactive_profile)


def test_create_load_batch_update(input_data, load_profiles):
    active_profile, reactive_profile = load_profiles

    batch_update = create_load_batch_update(active_profile, reactive_profile)

    assert validate_batch_data(input_data, batch_update) is None
    assert batch_update[ComponentType.sym_load]["id"][0].tolist() == active_profile.columns.tolist()
    np.testing.assert_array_equal(batch_update[ComponentType.sym_load]["p_specified"], active_profile.to_numpy())
    np.testing.assert_array_equal(batch_update[ComponentType.sym_load]["q_specified"], reactive_profile.to_numpy())


def test_run_batch_power_flow_invalid_update_raises_error(input_data, load_profiles):
    active_profile, reactive_profile = load_profiles
    batch_update = create_load_batch_update(active_profile, reactive_profile)
    batch_update[ComponentType.sym_load]["id"][0][0] = 999

    with pytest.raises(BatchDataValidationError, match="batch update"):
        run_batch_power_flow(input_data, batch_update)


def test_power_flow_and_aggregation_results(input_data, load_profiles):
    active_profile, reactive_profile = load_profiles
    batch_update = create_load_batch_update(active_profile, reactive_profile)

    results = run_batch_power_flow(input_data, batch_update)
    voltage_summary = aggregate_node_voltage_results(results, active_profile.index)
    line_summary = aggregate_line_results(results, active_profile.index)

    first_timestamp = active_profile.index[0]
    assert voltage_summary.loc[first_timestamp, "max_u_pu_node_id"] == 1
    assert voltage_summary.loc[first_timestamp, "min_u_pu_node_id"] == 3
    assert voltage_summary.loc[first_timestamp, "max_u_pu"] == pytest.approx(1.004847, rel=1e-5)
    assert voltage_summary.loc[first_timestamp, "min_u_pu"] == pytest.approx(1.003450, rel=1e-5)

    assert line_summary.loc[5, "max_loading_timestamp"] == active_profile.index[6]
    assert line_summary.loc[5, "min_loading_timestamp"] == active_profile.index[3]
    assert line_summary.loc[5, "max_loading_pu"] == pytest.approx(0.149830, rel=1e-5)
    assert line_summary.loc[5, "min_loading_pu"] == pytest.approx(0.063798, rel=1e-5)
    assert line_summary.loc[5, "energy_loss_kwh"] == pytest.approx(63.294763, rel=1e-5)


def test_aggregation_rejects_wrong_number_of_timestamps(input_data, load_profiles):
    active_profile, reactive_profile = load_profiles
    batch_update = create_load_batch_update(active_profile, reactive_profile)
    results = run_batch_power_flow(input_data, batch_update)
    incomplete_time_index = active_profile.index[:-1]

    with pytest.raises(TimeIndexLengthError):
        aggregate_node_voltage_results(results, incomplete_time_index)
    with pytest.raises(TimeIndexLengthError):
        aggregate_line_results(results, incomplete_time_index)
