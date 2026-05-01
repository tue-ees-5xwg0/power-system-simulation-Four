from power_system_simulation.graph_processing import GraphProcessor

gp = GraphProcessor(
    vertex_ids=[0, 1, 2],
    edge_ids=[10, 20],
    edge_vertex_id_pairs=[(0, 1), (1, 2)],
    edge_enabled=[True, True],
    source_vertex_id=0
)

print("Graph created successfully!")
