# training data loader

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

K_INPUT, K_ACTIONS = 7 * 8 * 8, 64 * 64
DT = np.dtype(
    [
        ("board_tensor", np.float32, (K_INPUT,)),
        ("legal_mask", np.bool, (K_ACTIONS,)),
        ("policy", np.float32, (K_ACTIONS,)),
        ("child_visit_counts", np.int32, (K_ACTIONS,)),
        ("child_values", np.float32, (K_ACTIONS,)),
        ("value", np.float32),
        ("final_value", np.int32),
    ]
)


class TrainingDataset(Dataset):
    def __init__(self, filename):
        self.memmap = np.memmap(filename, dtype=DT, mode="r")

    def __len__(self):
        return len(self.memmap)

    def __getitem__(self, idx):
        rec = self.memmap[idx]
        board_tensor = (
            torch.from_numpy(rec["board_tensor"].copy()).float().reshape(7, 8, 8)
        )
        legal_mask = torch.from_numpy(rec["legal_mask"].copy()).bool()
        policy = torch.from_numpy(rec["policy"].copy()).float()
        child_visit_counts = torch.from_numpy(rec["child_visit_counts"].copy()).int()
        child_values = torch.from_numpy(rec["child_values"].copy()).float()
        value = torch.tensor(rec["value"], dtype=torch.float32)
        final_value = torch.tensor(rec["final_value"], dtype=torch.int32)
        return {
            "board_tensor": board_tensor,
            "policy": policy,
            "legal_mask": legal_mask,
            "child_visit_counts": child_visit_counts,
            "child_values": child_values,
            "value": value,
            "final_value": final_value,
        }


# Usage example:
if __name__ == "__main__":
    dataset = TrainingDataset("training_data.bin")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Loaded {len(dataset)} training records.")
    sample = dataset[0]
    print(f"Board shape: {sample['board_tensor'].shape}.")
    print(f"Legal Mask shape: {sample['legal_mask'].shape}.")
    print(f"Policy shape: {sample['policy'].shape}.")
    print(f"Visits shape: {sample['child_visit_counts'].shape}.")
    print(f"Values shape: {sample['value'].shape}.")
    print(f"Finals shape: {sample['final_value'].shape}.")

    # Print out the first 5 records using the DataLoader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Board Tensor:\n{batch['board_tensor'][0]}")
        print(f"  Sum of board tensor: {batch['board_tensor'][0].sum().item()}")
        print(f"  Legal Mask:\n{batch['legal_mask'][0]}")
        print(f"  Number of legal moves: {batch['legal_mask'][0].sum().item()}")
        print(f"  Policy:\n{batch['policy'][0]}")
        print(f"  Policy vector total: {batch['policy'][0].sum().item()}")
        print(f"  Child Visit Counts:\n{batch['child_visit_counts'][0]}")
        print(
            f"  Total child visit counts: {batch['child_visit_counts'][0].sum().item()}"
        )
        print(f"  Value:\n{batch['value'][0]}")
        print(f"  Final Value:\n{batch['final_value'][0]}")
        if i >= 0:  # Only print the first batch for brevity
            break
