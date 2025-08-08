import numpy as np

K_INPUT, K_ACTIONS = 7*8*8, 64*64
DT = np.dtype([
    ("board_tensor",       np.float32, (K_INPUT,)),
    ("policy",             np.float32, (K_ACTIONS,)),
    ("child_visit_counts", np.int32,   (K_ACTIONS,)),
    ("value",              np.float32),
    ("final_value",        np.int32),
])

records = np.fromfile("training_data.bin", dtype=DT)

print(f"Loaded {len(records)} training records.")
print(f"Board shape: {records['board_tensor'].shape}.")
print(f"Policy shape: {records['policy'].shape}.")
print(f"Visits shape: {records['child_visit_counts'].shape}.")
print(f"Values shape: {records['value'].shape}.")
print(f"Finals shape: {records['final_value'].shape}.")
print(f"First board: {records['board_tensor'][0].reshape(7, 8, 8)}")
print(f"First policy: {records['policy'][0]}")
print(f"First visits: {records['child_visit_counts'][0]}")
print(f"First value: {records['value'][0]}")
print(f"First final value: {records['final_value'][0]}")
