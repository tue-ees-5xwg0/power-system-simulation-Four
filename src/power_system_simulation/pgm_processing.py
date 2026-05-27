from pathlib import Path

import pandas as pd
from power_grid_model import ComponentType, PowerGridModel
from power_grid_model.utils import json_deserialize


def read_pgm_json(file_path):
    data = Path(file_path).read_text()
    return json_deserialize(data)

def create_pgm(input_data):
    return PowerGridModel(input_data)

def read_load_profile(file_path):
    return pd.read_parquet(file_path)

def velidate_load_profile(active_profile, reactive_profile):
    if not active_profile.index.equals(reactive_profile.index):
        raise ValueError("Active and reactive profiles must have the same index.")
    if not active_profile.columns.equals(reactive_profile.columns):
        raise ValueError("Active and reactive profiles must have the same columns.")

def create_load_batch_update(active_profile, reactive_profile):
    velidate_load_profile(active_profile, reactive_profile)
    load_ids = active_profile.columns.to_numpy()

    return {
        ComponentType.sym_load: {
            "id": active_profile.columns.to_numpy(),
            "p_specified": active_profile.to_numpy(),
            "q_specified": reactive_profile.to_numpy(),
        }
    }


