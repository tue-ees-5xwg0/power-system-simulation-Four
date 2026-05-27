from power_system_simulation.pgm_processing import (
    ComponentType,
    create_load_batch_update,
    create_pgm,
    read_load_profile,
    read_pgm_json,
    velidate_load_profile,
)


def test_read_pgm_json():

    result = read_pgm_json("data/input_network_data.json")

    assert result is not None

input_data = read_pgm_json("data/input_network_data.json")
model = create_pgm(input_data)

print(model)

active_profile = read_load_profile("data/active_power_profile.parquet")

print(active_profile)
print(active_profile.index)
print(active_profile.columns)

def test_create_load_batch_update():
    active_profile = read_load_profile("data/active_power_profile.parquet")
    reactive_profile = read_load_profile("data/reactive_power_profile.parquet")
    velidate_load_profile(active_profile, reactive_profile)

    batch_update = create_load_batch_update(active_profile, reactive_profile)

    assert batch_update[ComponentType.sym_load]["id"].tolist() == active_profile.columns.tolist()
    assert batch_update[ComponentType.sym_load]["p_specified"].shape == active_profile.shape
    assert batch_update[ComponentType.sym_load]["q_specified"].shape == reactive_profile.shape
