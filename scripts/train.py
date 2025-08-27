import os
import re
import sys
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import TrainingDataset


# Redirect print statements to both console and log file
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        self.terminal.write(message)
        if self.log != "\n":
            self.log.write(timestamp)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger("training.log")


def get_next_model_path(out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)
    model_files = [f for f in os.listdir(out_dir) if re.match(r"model_(\d+)\.pt$", f)]
    numbers = [
        int(re.match(r"model_(\d+)\.pt$", f).group(1))
        for f in model_files
        if re.match(r"model_(\d+)\.pt$", f)
    ]
    next_num = max(numbers, default=0) + 1
    return os.path.join(out_dir, f"model_{next_num:03d}.pt")


def main():
    # Hyperparameters
    batch_size = 2048  # Back to working batch size
    lr = 5e-3  # High but stable learning rate
    l2_weight = 1e-4  # Standard regularization

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = TrainingDataset("training_data.bin")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(1, os.cpu_count() // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Load model
    model = torch.jit.load("model.pt").to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_weight)

    losses = []

    print("Starting training...")
    print(f"Dataset size = {len(dataset)} ({len(dataloader)} batches)")

    use_legal_mask = True

    loss_history = []
    start_time = time.time()
    timeout = 60  # 60 seconds
    steps = 0
    epoch = 0
    policy_loss_history = []
    value_loss_history = []
    batch_count = 0

    while time.time() - start_time < timeout:
        for i, batch in enumerate(dataloader):
            x = batch["board_tensor"].to(device)  # (B, 7, 8, 8)
            y1 = batch["child_visit_counts"].float().to(device)  # (B, 4096)
            y2 = batch["final_value"].float().to(device)  # (B,)
            legal_mask = batch["legal_mask"].to(device)  # (B, 4096), 0/1 or bool

            policy_logits, value_pred = model(x)  # (B, 4096), (B,)
            if use_legal_mask:
                policy_logits *= legal_mask
                # policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e10)

            value_loss = F.mse_loss(value_pred, y2)

            target_policy = y1 / (y1.sum(dim=1, keepdim=True) + 1e-6)

            log_probs = F.log_softmax(policy_logits, dim=1)

            log_target_policy = target_policy.clamp(min=1e-12).log()
            policy_loss = (
                (target_policy * (log_target_policy - log_probs)).sum(dim=1).mean()
            )

            loss = value_loss + policy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"\tBatch {i+1:4}\tLoss: {loss.item():.4f} (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})"
                )
            loss_history.append(loss.item())
            policy_loss_history.append(policy_loss.item())
            value_loss_history.append(value_loss.item())

            batch_count += 1
            steps += 1

            if time.time() - start_time >= timeout:
                break

        epoch += 1
        if time.time() - start_time >= timeout:
            break

    # After training, save loss plot
    plt.figure(figsize=(12, 6))
    batch_indices = list(range(0, len(loss_history) * 100, 100))
    plt.plot(
        batch_indices,
        loss_history,
        marker=".",
        label="Total Loss",
        color="blue",
        markersize=4,
    )
    plt.plot(
        batch_indices,
        policy_loss_history,
        marker=".",
        label="Policy Loss",
        color="orange",
        markersize=4,
    )
    plt.plot(
        batch_indices,
        value_loss_history,
        marker=".",
        label="Value Loss",
        color="red",
        markersize=4,
    )
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Training Losses (per 100 batches)")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.close()
    print("Saved loss plot to loss_plot.png")

    # Save model after training
    save_path = get_next_model_path("out")
    torch.jit.save(model, save_path)
    print(f"Model saved to {save_path}")

    # Also save as model.pt for next iteration
    torch.jit.save(model, "model.pt")
    print("Model also saved to model.pt")


if __name__ == "__main__":
    main()
