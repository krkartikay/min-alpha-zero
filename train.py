import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
import re

from dataset import TrainingDataset
import numpy as np

def masked_log_softmax(logits, mask, dim=-1):
    # Set logits to large negative where mask is False
    mask = mask.bool()
    logits = logits.masked_fill(~mask, float('-inf'))
    return F.log_softmax(logits, dim=dim)

def get_next_model_path(out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)
    model_files = [f for f in os.listdir(out_dir) if re.match(r"model_(\d+)\.pt$", f)]
    numbers = [int(re.match(r"model_(\d+)\.pt$", f).group(1)) for f in model_files if re.match(r"model_(\d+)\.pt$", f)]
    next_num = max(numbers, default=0) + 1
    return os.path.join(out_dir, f"model_{next_num:03d}.pt")

def main():
    # Hyperparameters
    batch_size = 32
    lr = 1e-3
    epochs = 5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = TrainingDataset("training_data.bin")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = torch.jit.load("model.pt").to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            board_tensor = batch['board_tensor'].to(device)  # (B, 7, 8, 8)
            legal_mask = batch['legal_mask'].to(device)      # (B, 4096)
            child_visit_counts = batch['child_visit_counts'].float().to(device)  # (B, 4096)
            final_value = batch['final_value'].float().to(device)  # (B,)

            # Forward pass
            policy_logits, value_pred = model(board_tensor)

            # Masked log softmax
            log_probs = masked_log_softmax(policy_logits, legal_mask, dim=1)

            # Normalize child_visit_counts to get target policy
            target_policy = child_visit_counts / (child_visit_counts.sum(dim=1, keepdim=True) + 1e-8)

            # Cross-entropy loss (negative log-likelihood)
            policy_loss = F.cross_entropy(policy_logits, target_policy, reduction='mean')

            # MSE loss for value prediction
            value_loss = F.mse_loss(value_pred.squeeze(-1), final_value)

            # Total loss
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * board_tensor.size(0)
            print(f"Batch {i+1} \t Policy loss: {policy_loss.item():.4f}\t"
                  f"Value loss: {value_loss.item():.4f}\t Total loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # Save model after training
    save_path = get_next_model_path("out")
    torch.jit.save(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
