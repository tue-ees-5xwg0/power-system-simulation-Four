import pandas as pd
import pytest
from power_grid_model import ComponentType, DatasetType, LoadGenType, WindingType, initialize_array


@pytest.fixture
def lv_grid():
    nodes = initialize_array(DatasetType.input, ComponentType.node, 6)
    nodes["id"] = [1, 3, 6, 7, 9, 12]
    nodes["u_rated"] = [10500, 400, 400, 400, 400, 400]

    source = initialize_array(DatasetType.input, ComponentType.source, 1)
    source["id"] = 0
    source["node"] = 1
    source["status"] = 1
    source["u_ref"] = 1.0
    source["sk"] = 100e6
    source["rx_ratio"] = 0.1

    transformer = initialize_array(DatasetType.input, ComponentType.transformer, 1)
    transformer["id"] = 2
    transformer["from_node"] = 1
    transformer["to_node"] = 3
    transformer["from_status"] = 1
    transformer["to_status"] = 1
    transformer["u1"] = 10500
    transformer["u2"] = 400
    transformer["sn"] = 400e3
    transformer["uk"] = 0.04
    transformer["pk"] = 4000
    transformer["i0"] = 0
    transformer["p0"] = 0
    transformer["winding_from"] = WindingType.delta
    transformer["winding_to"] = WindingType.wye_n
    transformer["clock"] = 5
    transformer["tap_side"] = 0
    transformer["tap_pos"] = 0
    transformer["tap_min"] = -1
    transformer["tap_max"] = 1
    transformer["tap_nom"] = 0
    transformer["tap_size"] = 100

    lines = initialize_array(DatasetType.input, ComponentType.line, 5)
    lines["id"] = [4, 5, 8, 11, 24]
    lines["from_node"] = [3, 3, 6, 7, 9]
    lines["to_node"] = [6, 7, 9, 12, 12]
    lines["from_status"] = 1
    lines["to_status"] = [1, 1, 1, 1, 0]
    lines["r1"] = 0.05
    lines["x1"] = 0.01
    lines["c1"] = 0
    lines["tan1"] = 0
    lines["i_n"] = 200

    loads = initialize_array(DatasetType.input, ComponentType.sym_load, 2)
    loads["id"] = [10, 13]
    loads["node"] = [9, 12]
    loads["status"] = 1
    loads["type"] = LoadGenType.const_power
    loads["p_specified"] = 0
    loads["q_specified"] = 0

    return {
        ComponentType.node: nodes,
        ComponentType.source: source,
        ComponentType.transformer: transformer,
        ComponentType.line: lines,
        ComponentType.sym_load: loads,
    }


@pytest.fixture
def lv_profiles():
    index = pd.date_range("2026-01-01", periods=3, freq="h")
    load_p = pd.DataFrame({10: [2000.0, 3000.0, 2500.0], 13: [2000.0, 2500.0, 3000.0]}, index=index)
    load_q = pd.DataFrame({10: [500.0, 600.0, 550.0], 13: [500.0, 550.0, 600.0]}, index=index)
    ev_profiles = pd.DataFrame({"ev_a": [0.0, 1000.0, 500.0], "ev_b": [500.0, 0.0, 1000.0]}, index=index)
    return load_p, load_q, ev_profiles
