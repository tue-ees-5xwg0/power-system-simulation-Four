import numpy as np
import pandas as pd
import pytest
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import ValidationException

from power_system_simulation.power_grid_calculation import PowerGridCalculation, ProfileDoesNotMatchError

DATA = "input/input/"
EXPECTED = "expected_output/expected_output/"


@pytest.fixture
def pgm_data():
    with open(DATA + "input_network_data.json") as f:
        return json_deserialize(f.read())


@pytest.fixture
def active():
    return pd.read_parquet(DATA + "active_power_profile.parquet")


@pytest.fixture
def reactive():
    return pd.read_parquet(DATA + "reactive_power_profile.parquet")


@pytest.fixture
def calc(pgm_data, active, reactive):
    return PowerGridCalculation(pgm_data, active, reactive)


def test_voltage_table(calc):
    result = calc.voltage_table()
    expected = pd.read_parquet(EXPECTED + "output_table_row_per_timestamp.parquet")
    np.testing.assert_allclose(result["Max_Voltage"].values, expected["Max_Voltage"].values)
    np.testing.assert_allclose(result["Min_Voltage"].values, expected["Min_Voltage"].values)
    assert list(result["Max_Voltage_Node"]) == list(expected["Max_Voltage_Node"])
    assert list(result["Min_Voltage_Node"]) == list(expected["Min_Voltage_Node"])


def test_line_table(calc):
    result = calc.line_table()
    expected = pd.read_parquet(EXPECTED + "output_table_row_per_line.parquet")
    np.testing.assert_allclose(result["Total_Loss"].values, expected["Total_Loss"].values)
    np.testing.assert_allclose(result["Max_Loading"].values, expected["Max_Loading"].values)
    np.testing.assert_allclose(result["Min_Loading"].values, expected["Min_Loading"].values)
    assert list(result["Max_Loading_Timestamp"]) == list(expected["Max_Loading_Timestamp"])
    assert list(result["Min_Loading_Timestamp"]) == list(expected["Min_Loading_Timestamp"])


def test_mismatched_timestamps(pgm_data, active, reactive):
    bad_reactive = reactive.copy()
    bad_reactive.index = pd.date_range("2025-01-01", periods=len(reactive), freq="h")
    with pytest.raises(ProfileDoesNotMatchError):
        PowerGridCalculation(pgm_data, active, bad_reactive)


def test_mismatched_load_ids(pgm_data, active, reactive):
    bad_reactive = reactive.copy()
    bad_reactive.columns = [999, 998, 997]
    with pytest.raises(ProfileDoesNotMatchError):
        PowerGridCalculation(pgm_data, active, bad_reactive)


def test_invalid_input_data(active, reactive):
    with pytest.raises((KeyError, TypeError, ValueError)):
        PowerGridCalculation({"node": [], "line": []}, active, reactive)


def test_invalid_batch_data(pgm_data, active, reactive):
    bad_active = active.copy()
    bad_reactive = reactive.copy()
    bad_active.columns = [999, 998, 997]
    bad_reactive.columns = [999, 998, 997]
    with pytest.raises(ValidationException):
        PowerGridCalculation(pgm_data, bad_active, bad_reactive)