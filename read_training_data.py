import os
import numpy as np

K_INPUT = 7*8*8
K_ACTIONS = 64*64


def load_training_file(path, k_input_size: int, k_num_actions: int, *, as_torch: bool = False, mmap: bool = False):
    """
    Reads records written by append_to_training_file().

    Each record = {
      board_tensor: float32[k_input_size],
      policy: float32[k_num_actions],
      child_visit_counts: int32[k_num_actions],
      value: float32,
      final_value: int32
    }
    """
    dt = np.dtype([
        ("board_tensor",        np.float32, (k_input_size,)),
        ("policy",              np.float32, (k_num_actions,)),
        ("child_visit_counts",  np.int32,   (k_num_actions,)),
        ("value",               np.float32),
        ("final_value",         np.int32),
    ])

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # # Validate file size is a multiple of one record
    # fsize = os.path.getsize(path)
    # if fsize % dt.itemsize != 0:
    #     raise ValueError(f"File size {fsize} not multiple of record size {dt.itemsize} — file may be truncated/corrupt.")

    loader = np.memmap if mmap else np.fromfile
    recs = loader(path, dtype=dt)

    # Expose as simple dict of arrays
    out = {
        "boards":  recs["board_tensor"],        # (N, k_input_size) float32
        "policy":  recs["policy"],              # (N, k_num_actions) float32
        "visits":  recs["child_visit_counts"],  # (N, k_num_actions) int32
        "value":   recs["value"],               # (N,) float32 (model-pred)
        "final":   recs["final_value"],         # (N,) int32   (game result)
    }

    return out

data = load_training_file("training_data.bin", k_input_size=K_INPUT, k_num_actions=K_ACTIONS, as_torch=False)
boards  = data["boards"]   # (N, K_INPUT) float32
policy  = data["policy"]   # (N, K_ACTIONS) float32
visits  = data["visits"]   # (N, K_ACTIONS) int32
values  = data["value"]    # (N,) float32
finals  = data["final"]    # (N,) int32

print(f"Loaded {len(boards)} training records.")
print(f"Board shape: {boards.shape}.")
print(f"Policy shape: {policy.shape}.")
print(f"Visits shape: {visits.shape}.")
print(f"Values shape: {values.shape}.")
print(f"Finals shape: {finals.shape}.")
print(f"First board: {boards[0].reshape(7, 8, 8)}")
print(f"First policy: {policy[0]}")
print(f"First visits: {visits[0]}")
print(f"First value: {values[0]}")
print(f"First final: {finals[0]}")