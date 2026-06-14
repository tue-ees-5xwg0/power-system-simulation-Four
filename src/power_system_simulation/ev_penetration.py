import random

from power_grid_model import ComponentType

from .lv_grid_validation import validate_ev_profiles, validate_feeders, validate_grid, validate_load_profiles
from .time_series_analysis import run_time_series_power_flow


def apply_ev_penetration(grid, feeder_ids, load_p, load_q, ev_profiles, penetration, seed=None):
    graph = validate_grid(grid)
    validate_feeders(grid, feeder_ids)
    validate_load_profiles(grid, load_p, load_q)
    validate_ev_profiles(grid, load_p, ev_profiles)

    if not 0 <= penetration <= 1:
        raise ValueError("EV penetration must be between 0 and 1.")

    rng = random.Random(seed)
    loads = grid[ComponentType.sym_load]
    load_node = dict(zip(loads["id"], loads["node"], strict=True))
    ev_per_feeder = int(penetration * len(load_node) / len(feeder_ids))

    updated_load_p = load_p.copy()
    available_profiles = list(ev_profiles.columns)
    rng.shuffle(available_profiles)

    profile_index = 0
    for feeder_id in feeder_ids:
        downstream_nodes = set(graph.find_downstream_vertices(feeder_id))
        feeder_loads = [load_id for load_id, node_id in load_node.items() if node_id in downstream_nodes]
        if len(feeder_loads) < ev_per_feeder:
            raise ValueError("A feeder does not contain enough houses for this EV penetration.")

        selected_loads = rng.sample(feeder_loads, ev_per_feeder)
        for load_id in selected_loads:
            profile = available_profiles[profile_index]
            updated_load_p[load_id] += ev_profiles[profile]
            profile_index += 1

    return run_time_series_power_flow(grid, updated_load_p, load_q)
