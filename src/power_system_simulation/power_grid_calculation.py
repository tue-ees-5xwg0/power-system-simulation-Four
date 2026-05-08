import numpy as np
import pandas as pd
from power_grid_model import PowerGridModel, initialize_array
from power_grid_model.validation import assert_valid_batch_data, assert_valid_input_data


class ProfileDoesNotMatchError(Exception):
    pass


class PowerGridCalculation:
    def __init__(self, pgm_input_data, active_load_profile, reactive_load_profile):
        if not active_load_profile.index.equals(reactive_load_profile.index):
            raise ProfileDoesNotMatchError("Timestamps do not match.")
        if list(active_load_profile.columns) != list(reactive_load_profile.columns):
            raise ProfileDoesNotMatchError("Load IDs do not match.")

        assert_valid_input_data(pgm_input_data, symmetric=True)
        model = PowerGridModel(pgm_input_data)

        self._timestamps = active_load_profile.index
        n_batch, n_loads = active_load_profile.shape

        update = initialize_array("update", "sym_load", (n_batch, n_loads))
        update["id"] = list(active_load_profile.columns)
        update["p_specified"] = active_load_profile.values
        update["q_specified"] = reactive_load_profile.values
        batch = {"sym_load": update}

        assert_valid_batch_data(pgm_input_data, batch, symmetric=True)
        self._output = model.calculate_power_flow(update_data=batch, symmetric=True)

    def voltage_table(self):
        node_ids = self._output["node"]["id"][0]
        u_pu = self._output["node"]["u_pu"]
        n = np.arange(len(self._timestamps))
        max_i = np.argmax(u_pu, axis=1)
        min_i = np.argmin(u_pu, axis=1)

        return pd.DataFrame({
            "Max_Voltage": u_pu[n, max_i],
            "Max_Voltage_Node": node_ids[max_i],
            "Min_Voltage": u_pu[n, min_i],
            "Min_Voltage_Node": node_ids[min_i],
        }, index=self._timestamps)

    def line_table(self):
        line_ids = self._output["line"]["id"][0]
        loading = self._output["line"]["loading"]
        p_loss = self._output["line"]["p_from"] + self._output["line"]["p_to"]

        t_s = self._timestamps.astype(np.int64).values / 1e9
        energy_kwh = np.trapezoid(p_loss, t_s, axis=0) / 3_600_000

        n = len(line_ids)
        max_i = np.argmax(loading, axis=0)
        min_i = np.argmin(loading, axis=0)

        return pd.DataFrame({
            "Total_Loss": energy_kwh,
            "Max_Loading": loading[max_i, np.arange(n)],
            "Max_Loading_Timestamp": self._timestamps[max_i],
            "Min_Loading": loading[min_i, np.arange(n)],
            "Min_Loading_Timestamp": self._timestamps[min_i],
        }, index=line_ids)
