from pathlib import Path

import numpy as np
import pandas as pd
from power_grid_model import ComponentType, PowerGridModel
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import errors_to_string, validate_batch_data, validate_input_data


class InputDataValidationError(ValueError):
    pass


class BatchDataValidationError(ValueError):
    pass


class ProfileTimestampMismatchError(ValueError):
    pass


class ProfileLoadIDMismatchError(ValueError):
    pass


class TimeIndexLengthError(ValueError):
    pass


def read_json_file(file_path):
    return Path(file_path).read_text()


def deserialize_pgm_json(json_data):
    return json_deserialize(json_data)


def create_pgm(input_data):
    errors = validate_input_data(input_data)
    if errors:
        raise InputDataValidationError(errors_to_string(errors, name="input data", details=True))
    return PowerGridModel(input_data)


def read_load_profile(file_path):
    return pd.read_parquet(file_path)


def validate_load_profile(active_profile, reactive_profile):
    if not active_profile.index.equals(reactive_profile.index):
        raise ProfileTimestampMismatchError("Active and reactive profiles have different timestamps.")
    if not active_profile.columns.equals(reactive_profile.columns):
        raise ProfileLoadIDMismatchError("Active and reactive profiles have different load IDs.")


def create_load_batch_update(active_profile, reactive_profile):
    validate_load_profile(active_profile, reactive_profile)

    load_ids = np.tile(
        active_profile.columns.to_numpy(),
        (len(active_profile), 1),
    )

    return {
        ComponentType.sym_load: {
            "id": load_ids,
            "p_specified": active_profile.to_numpy(),
            "q_specified": reactive_profile.to_numpy(),
        }
    }


def run_batch_power_flow(input_data, batch_update):
    errors = validate_batch_data(input_data, batch_update)
    if errors:
        raise BatchDataValidationError(errors_to_string(errors, name="batch update", details=True))
    model = PowerGridModel(input_data)
    return model.calculate_power_flow(update_data=batch_update)


def aggregate_node_voltage_results(results, time_index):
    node_results = results[ComponentType.node]
    if len(node_results) != len(time_index):
        raise TimeIndexLengthError("The number of timestamps does not match the number of result batches.")

    rows = []
    for i, timestamp in enumerate(time_index):
        ids = node_results[i]["id"]
        u_pu = node_results[i]["u_pu"]

        max_idx = np.argmax(u_pu)
        min_idx = np.argmin(u_pu)

        rows.append(
            {
                "timestamp": timestamp,
                "max_u_pu": u_pu[max_idx],
                "max_u_pu_node_id": ids[max_idx],
                "min_u_pu": u_pu[min_idx],
                "min_u_pu_node_id": ids[min_idx],
            }
        )
    return pd.DataFrame(rows).set_index("timestamp")


def aggregate_line_results(results, time_index):
    line_results = results[ComponentType.line]
    if len(line_results) != len(time_index):
        raise TimeIndexLengthError("The number of timestamps does not match the number of result batches.")

    line_ids = line_results[0]["id"]

    rows = []
    for line_index, line_id in enumerate(line_ids):
        loading = line_results["loading"][:, line_index]
        max_idx = np.argmax(loading)
        min_idx = np.argmin(loading)
        loss_w = line_results["p_from"][:, line_index] + line_results["p_to"][:, line_index]
        hours = (time_index - time_index[0]).total_seconds() / 3600
        energy_loss_kwh = np.trapezoid(loss_w, hours) / 1000
        rows.append(
            {
                "line_id": line_id,
                "max_loading_pu": loading[max_idx],
                "max_loading_timestamp": time_index[max_idx],
                "min_loading_pu": loading[min_idx],
                "min_loading_timestamp": time_index[min_idx],
                "energy_loss_kwh": energy_loss_kwh,
            }
        )

    return pd.DataFrame(rows).set_index("line_id")
