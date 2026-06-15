import copy
import random

import pandas as pd

from .graph_processing import GraphProcessor

# IMPORTANT: this must exist so tests can monkeypatch it
from .pgm_processing import (
    aggregate_line_results,
    aggregate_node_voltage_results,
    create_load_batch_update,
    create_pgm,
    run_batch_power_flow,
)


# --------------------------------------------------
# Helper function (this is what tests patch!)
# --------------------------------------------------
def run_time_series_power_flow(grid, load_p, load_q):
    model = create_pgm(grid)

    batch_update = create_load_batch_update(load_p, load_q)
    results = run_batch_power_flow(model, batch_update)

    voltage_table = aggregate_node_voltage_results(results, load_p.index)
    line_table = aggregate_line_results(results, load_p.index)

    return voltage_table, line_table


# --------------------------------------------------
# MAIN CLASS
# --------------------------------------------------
class LVGridAnalysis:
    def __init__(
        self,
        grid_data,
        feeders,
        load_p: pd.DataFrame,
        load_q: pd.DataFrame,
        ev_profiles: pd.DataFrame,
    ):
        self.grid = grid_data
        self.feeders = feeders
        self.load_p = load_p.copy()
        self.load_q = load_q.copy()
        self.ev_profiles = ev_profiles

        self._validate_inputs()
        self.graph_processor = self._create_graph_processor()

    # ==================================================
    # 1. INPUT VALIDATION
    # ==================================================
    def _validate_inputs(self):
        if len(self.grid["transformers"]) != 1:
            raise ValueError("Grid must have exactly one transformer")

        if len(self.grid["sources"]) != 1:
            raise ValueError("Grid must have exactly one source")

        line_ids = {l["id"] for l in self.grid["lines"]}  # noqa: E741
        for f in self.feeders:
            if f not in line_ids:
                raise ValueError(f"Invalid feeder ID {f}")

        if not self.load_p.index.equals(self.load_q.index):
            raise ValueError("Timestamps mismatch")

        if set(self.load_p.columns) != set(self.load_q.columns):
            raise ValueError("Load IDs mismatch")

        if self.ev_profiles.shape[1] < len(self.load_p.columns):
            raise ValueError("Not enough EV profiles")

    # ==================================================
    # 2. GRAPH (Assignment 1 reuse)
    # ==================================================
    def _create_graph_processor(self):
        vertex_ids = [n["id"] for n in self.grid["nodes"]]

        edge_ids = []
        edge_pairs = []
        edge_enabled = []

        for line in self.grid["lines"]:
            edge_ids.append(line["id"])
            edge_pairs.append((line["from_node"], line["to_node"]))

            enabled = line["from_status"] == 1 and line["to_status"] == 1
            edge_enabled.append(enabled)

        source = self.grid["sources"][0]["node"]

        return GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_pairs,
            edge_enabled,
            source,
        )

    # ==================================================
    # 3. EV PENETRATION
    # ==================================================
    def apply_ev_penetration(self, penetration, seed=None):
        if seed is not None:
            random.seed(seed)

        houses = list(self.load_p.columns)
        n_houses = len(houses)
        n_feeders = len(self.feeders)

        ev_per_feeder = int((penetration * n_houses) // n_feeders)

        available_profiles = list(self.ev_profiles.columns)
        random.shuffle(available_profiles)

        profile_idx = 0

        for feeder in self.feeders:
            downstream = self.graph_processor.find_downstream_vertices(feeder)

            feeder_houses = [h for h in downstream if h in houses]

            selected = random.sample(
                feeder_houses,
                min(ev_per_feeder, len(feeder_houses)),
            )

            for house in selected:
                profile = available_profiles[profile_idx]
                profile_idx += 1

                self.load_p[house] += self.ev_profiles[profile]

        return run_time_series_power_flow(self.grid, self.load_p, self.load_q)

    # ==================================================
    # 4. TAP OPTIMIZATION
    # ==================================================
    def optimize_tap(self, tap_positions, criterion="loss"):
        transformer = self.grid["transformers"][0]

        best_tap = None
        best_value = float("inf")

        for tap in tap_positions:
            transformer["tap_position"] = tap

            voltage_table, line_table = run_time_series_power_flow(
                self.grid,
                self.load_p,
                self.load_q,
            )

            if criterion == "loss":
                value = line_table["energy_loss_kwh"].sum()

            elif criterion == "voltage":
                deviation = (
                    (voltage_table["max_u_pu"] - 1).abs()
                    + (voltage_table["min_u_pu"] - 1).abs()
                )
                value = deviation.mean()

            else:
                raise ValueError("Invalid criterion")

            if value < best_value:
                best_value = value
                best_tap = tap

        return best_tap

    # ==================================================
    # 5. N-1 ANALYSIS
    # ==================================================
    def n_minus_1(self, outage_line_id):
        alternatives = self.graph_processor.find_alternative_edges(outage_line_id)

        results = []

        for alt in alternatives:
            grid_copy = copy.deepcopy(self.grid)

            # Disconnect outage
            for line in grid_copy["lines"]:
                if line["id"] == outage_line_id:
                    line["from_status"] = 0
                    line["to_status"] = 0

            # Connect alternative
            for line in grid_copy["lines"]:
                if line["id"] == alt:
                    line["from_status"] = 1
                    line["to_status"] = 1

            voltage_table, line_table = run_time_series_power_flow(
                grid_copy,
                self.load_p,
                self.load_q,
            )

            # Find maximum loading
            idx = line_table["max_loading_pu"].idxmax()

            results.append({
                "alternative_line": alt,
                "max_loading": line_table.loc[idx, "max_loading_pu"],
                "line_id": idx,
                "timestamp": line_table.loc[idx, "max_loading_timestamp"],
            })

        return pd.DataFrame(results)
