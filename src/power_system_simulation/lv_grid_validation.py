import pandas as pd
from power_grid_model import ComponentType

from .graph_processing import GraphProcessor
from .pgm_processing import create_pgm, validate_load_profile


def create_line_graph(grid):
    lines = grid[ComponentType.line]
    edge_table = pd.DataFrame(
        {
            "edge_id": lines["id"],
            "from_vertex": lines["from_node"],
            "to_vertex": lines["to_node"],
            "enabled": (lines["from_status"] == 1) & (lines["to_status"] == 1),
        }
    )
    source_node = grid[ComponentType.transformer]["to_node"][0]
    return GraphProcessor(edge_table, source_vertex_id=source_node)


def validate_grid(grid):
    create_pgm(grid)

    transformers = grid.get(ComponentType.transformer)
    sources = grid.get(ComponentType.source)
    lines = grid.get(ComponentType.line)
    loads = grid.get(ComponentType.sym_load)

    if transformers is None or len(transformers) != 1:
        raise ValueError("The grid must contain exactly one transformer.")
    if sources is None or len(sources) != 1:
        raise ValueError("The grid must contain exactly one source.")
    if lines is None:
        raise ValueError("The grid must contain lines.")
    if loads is None:
        raise ValueError("The grid must contain symmetric loads.")

    return create_line_graph(grid)


def validate_feeders(grid, feeder_ids):
    if not feeder_ids:
        raise ValueError("At least one feeder ID is required.")

    lines = grid[ComponentType.line]
    line_ids = set(lines["id"].tolist())
    if not set(feeder_ids).issubset(line_ids):
        raise ValueError("All feeder IDs must be valid line IDs.")

    transformer_to_node = grid[ComponentType.transformer]["to_node"][0]
    for feeder_id in feeder_ids:
        feeder = lines[lines["id"] == feeder_id][0]
        if feeder["from_node"] != transformer_to_node:
            raise ValueError("Every feeder must start at the transformer to_node.")


def validate_load_profiles(grid, load_p, load_q):
    validate_load_profile(load_p, load_q)

    load_ids = set(grid[ComponentType.sym_load]["id"].tolist())
    if set(load_p.columns) != load_ids:
        raise ValueError("The load profile columns must match all sym_load IDs.")


def validate_ev_profiles(grid, load_p, ev_profiles):
    if not ev_profiles.index.equals(load_p.index):
        raise ValueError("EV profiles must have the same timestamps as the load profiles.")
    if ev_profiles.shape[1] < len(grid[ComponentType.sym_load]):
        raise ValueError("There must be at least one EV profile per sym_load.")
