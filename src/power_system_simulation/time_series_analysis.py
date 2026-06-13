from .pgm_processing import (
    aggregate_line_results,
    aggregate_node_voltage_results,
    create_load_batch_update,
    run_batch_power_flow,
)


def run_time_series_power_flow(grid, load_p, load_q):
    batch_update = create_load_batch_update(load_p, load_q)
    results = run_batch_power_flow(grid, batch_update)
    voltage_table = aggregate_node_voltage_results(results, load_p.index)
    line_table = aggregate_line_results(results, load_p.index)
    return voltage_table, line_table
