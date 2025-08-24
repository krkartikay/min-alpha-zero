import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import re
import sys  # Added for logging
from datetime import datetime  # Added for timestamps


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

from dataset import TrainingDataset
import numpy as np


def masked_log_softmax(logits, mask, dim=-1):
    # Set logits to large negative where mask is False
    mask = mask.bool()
    logits = logits.masked_fill(~mask, -1e10)
    return F.log_softmax(logits, dim=dim)


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
    batch_size = 64
    lr = 3e-4
    epochs = 5
    l2_weight = 1e-4  # L2 regularization weight
    min_improvement = 0.0001  # Minimum improvement to continue training

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
    epoch = 0
    prev_loss = float("inf")
    while True:
        epoch += 1
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            board_tensor = batch["board_tensor"].to(device)  # (B, 7, 8, 8)
            legal_mask = batch["legal_mask"].to(device)  # (B, 4096)
            child_visit_counts = (
                batch["child_visit_counts"].float().to(device)
            )  # (B, 4096)
            final_value = batch["final_value"].float().to(device)  # (B,)

            # Forward pass
            policy_logits, value_pred = model(board_tensor)

            # Masked log softmax
            masked_log_probs = masked_log_softmax(policy_logits, legal_mask, dim=1)

            # Normalize child_visit_counts to get target policy
            child_visit_counts_sum = child_visit_counts.sum(dim=1, keepdim=True)
            valid_rows = (child_visit_counts_sum > 0).squeeze(1)

            # Normalize only valid rows
            target_policy = torch.zeros_like(child_visit_counts)
            target_policy[valid_rows] = (
                child_visit_counts[valid_rows] / child_visit_counts_sum[valid_rows]
            )

            # Compute per-row loss
            # Policy loss: KL(target || model) == -sum target * log_probs + const
            row_loss = -(masked_log_probs * target_policy).sum(dim=1)

            # Average only over valid rows
            policy_loss = row_loss[valid_rows].mean()

            # MSE loss for value prediction
            value_loss = F.mse_loss(value_pred.squeeze(-1), final_value)

            # Total loss
            loss = policy_loss + value_loss

            # Assert target policy is positive only where legal mask is true
            assert (
                target_policy[~legal_mask] == 0
            ).all(), "Target policy must be positive only where legal mask is true"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * board_tensor.size(0)
            if (i + 1) % 100 == 0:
                print(
                    f"Batch {i+1} \t Policy loss: {policy_loss.item():.4f}\t"
                    f"Value loss: {value_loss.item():.4f}\t Total loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Accumulate epoch losses
        losses.append(float(avg_loss))
        # Check improvement vs. previous epoch
        if prev_loss < float("inf"):
            improvement = (prev_loss - avg_loss) / prev_loss
            print(f"Loss Improvement: {improvement:.2%}")
            if improvement < min_improvement:
                print(f"Stopping: improvement {improvement:.2%} < {min_improvement:.2%}")
                break
        prev_loss = avg_loss

    # After training, save loss plot
    plt.figure()
    epochs_range = list(range(1, len(losses) + 1))
    plt.plot(epochs_range, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Training Loss")
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
