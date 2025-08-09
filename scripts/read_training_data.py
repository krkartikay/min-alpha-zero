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

# Print out the records one by one
for i, record in enumerate(records):
    print(f"Record {i}:")
    print(f"  Board Tensor:\n{record['board_tensor'].reshape(7, 8, 8)}")
    print(f"  Policy:\n{record['policy']}")
    print(f"  Child Visit Counts:\n{record['child_visit_counts']}")
    print(f"  Value:\n{record['value']}")
    print(f"  Final Value:\n{record['final_value']}")
    print()  # Add a newline for better readability
    if i >= 5:  # Limit output to first 5 records for brevity
        break