import os
import re
import sys
import torch
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
    batch_size = 512
    lr = 1e-3
    l2_weight = 1e-4  # L2 regularization weight

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
    total_steps = 2000  # 2k batches... roughly takes 60s with current model
    steps = 0
    epoch = 0
    policy_loss_history = []
    value_loss_history = []

    while steps < total_steps:
        total_loss = torch.tensor(0.).to(device)
        total_policy_loss = torch.tensor(0.).to(device)
        total_value_loss = torch.tensor(0.).to(device)
        for i, batch in enumerate(dataloader):
            x = batch['board_tensor'].to(device)                    # (B, 7, 8, 8)
            y1 = batch['child_visit_counts'].float().to(device)     # (B, 4096)
            y2 = batch['final_value'].float().to(device)            # (B,)
            legal_mask = batch['legal_mask'].to(device)             # (B, 4096), 0/1 or bool

            policy_logits, value_pred = model(x) # (B, 4096), (B,)
            if use_legal_mask:
                policy_logits *= legal_mask

            value_loss = F.mse_loss(value_pred, y2)

            target_policy = y1 / (y1.sum(dim=1, keepdim=True)  + 1e-6)

            log_probs = F.log_softmax(policy_logits, dim=1)

            log_target_policy = target_policy.clamp(min=1e-12).log()
            policy_loss = (target_policy * (log_target_policy - log_probs)).sum(dim=1).mean()

            loss = value_loss + policy_loss
            total_loss += loss * x.shape[0]
            total_policy_loss += policy_loss * x.shape[0]
            total_value_loss += value_loss * x.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"\tBatch {i:4}\tLoss: {loss.item():.4f} (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})")
            steps += 1

        total_loss /= len(dataset)
        total_policy_loss /= len(dataset)
        total_value_loss /= len(dataset)
        loss_history.append(total_loss.item())
        policy_loss_history.append(total_policy_loss.item())
        value_loss_history.append(total_value_loss.item())
        epoch += 1
        print(f"Epoch {epoch:2} Step {steps:4}\tLoss: {total_loss.item():.4f} (Policy: {total_policy_loss.item():.4f}, Value: {total_value_loss.item():.4f})")

        if steps >= total_steps:
            break

    # After training, save loss plot
    plt.figure(figsize=(10, 6))
    epochs_range = list(range(1, len(loss_history) + 1))
    plt.plot(epochs_range, loss_history, marker="o", label="Total Loss", color="blue")
    plt.plot(epochs_range, policy_loss_history, marker="o", label="Policy Loss", color="orange")
    plt.plot(epochs_range, value_loss_history, marker="o", label="Value Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
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
