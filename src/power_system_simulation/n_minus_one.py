import copy

import numpy as np
import pandas as pd
from power_grid_model import ComponentType

from .lv_grid_validation import validate_grid, validate_load_profiles
from .time_series_analysis import run_time_series_power_flow

N_MINUS_1_COLUMNS = ["alternative_line_id", "max_loading_pu", "line_id", "timestamp"]


def calculate_n_minus_1(grid, load_p, load_q, outage_line_id):
    graph = validate_grid(grid)
    validate_load_profiles(grid, load_p, load_q)
    alternatives = graph.find_alternative_edges(outage_line_id)
    rows = []

    for alternative_line_id in alternatives:
        changed_grid = copy.deepcopy(grid)
        lines = changed_grid[ComponentType.line]

        outage_index = np.where(lines["id"] == outage_line_id)[0][0]
        lines["from_status"][outage_index] = 0
        lines["to_status"][outage_index] = 0

        alternative_index = np.where(lines["id"] == alternative_line_id)[0][0]
        lines["to_status"][alternative_index] = 1

        _, line_table = run_time_series_power_flow(changed_grid, load_p, load_q)
        max_line_id = line_table["max_loading_pu"].idxmax()
        rows.append(
            {
                "alternative_line_id": alternative_line_id,
                "max_loading_pu": line_table.loc[max_line_id, "max_loading_pu"],
                "line_id": max_line_id,
                "timestamp": line_table.loc[max_line_id, "max_loading_timestamp"],
            }
        )

    return pd.DataFrame(rows, columns=N_MINUS_1_COLUMNS)
