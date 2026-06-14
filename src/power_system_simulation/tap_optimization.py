import copy

import numpy as np
from power_grid_model import ComponentType

from .lv_grid_validation import validate_grid, validate_load_profiles
from .pgm_processing import aggregate_line_results, create_load_batch_update, run_batch_power_flow


def optimize_tap_position(grid, load_p, load_q, criterion="loss"):
    validate_grid(grid)
    validate_load_profiles(grid, load_p, load_q)

    if criterion not in {"loss", "voltage"}:
        raise ValueError("Criterion must be 'loss' or 'voltage'.")

    transformer = grid[ComponentType.transformer][0]
    best_tap = None
    best_value = np.inf

    for tap_position in range(int(transformer["tap_min"]), int(transformer["tap_max"]) + 1):
        changed_grid = copy.deepcopy(grid)
        changed_grid[ComponentType.transformer]["tap_pos"][0] = tap_position

        batch_update = create_load_batch_update(load_p, load_q)
        results = run_batch_power_flow(changed_grid, batch_update)

        if criterion == "loss":
            line_table = aggregate_line_results(results, load_p.index)
            value = line_table["energy_loss_kwh"].sum()
        else:
            node_voltages = results[ComponentType.node]["u_pu"]
            max_deviation = np.abs(node_voltages.max(axis=0) - 1)
            min_deviation = np.abs(node_voltages.min(axis=0) - 1)
            value = np.mean((max_deviation + min_deviation) / 2)

        if value < best_value:
            best_value = value
            best_tap = tap_position

    return best_tap
