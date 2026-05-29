from pathlib import Path

import numpy as np
import pandas as pd

try:
    from power_grid_model import ComponentType, PowerGridModel
    from power_grid_model.utils import json_deserialize
except ImportError:
    from .power_grid_model import ComponentType, PowerGridModel
    from .power_grid_model.utils import json_deserialize


def read_pgm_json(file_path):
    data = Path(file_path).read_text()
    return json_deserialize(data)


def create_pgm(input_data):
    return PowerGridModel(input_data)


def read_load_profile(file_path):
    return pd.read_parquet(file_path)


def validate_load_profile(active_profile, reactive_profile):
    if not active_profile.index.equals(reactive_profile.index):
        raise ValueError("Active and reactive profiles must have the same index.")
    if not active_profile.columns.equals(reactive_profile.columns):
        raise ValueError("Active and reactive profiles must have the same columns.")


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


def run_batch_power_flow(model, batch_update):
    return model.calculate_power_flow(update_data=batch_update)


def aggregate_node_voltage_results(results, time_index):
    node_results = results[ComponentType.node]

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
