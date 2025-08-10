import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataset import TrainingDataset
import numpy as np

def masked_log_softmax(logits, mask, dim=-1):
    # Set logits to large negative where mask is False
    mask = mask.bool()
    logits = logits.masked_fill(~mask, float('-inf'))
    return F.log_softmax(logits, dim=dim)

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
        for batch in dataloader:
            board_tensor = batch['board_tensor'].to(device)  # (B, 7, 8, 8)
            legal_mask = batch['legal_mask'].to(device)      # (B, 4096)
            child_visit_counts = batch['child_visit_counts'].float().to(device)  # (B, 4096)

            # Forward pass
            policy_logits, _ = model(board_tensor)

            # Masked log softmax
            log_probs = masked_log_softmax(policy_logits, legal_mask, dim=1)

            # Normalize child_visit_counts to get target policy
            target_policy = child_visit_counts / (child_visit_counts.sum(dim=1, keepdim=True) + 1e-8)

            # Cross-entropy loss (negative log-likelihood)
            loss = -(target_policy * log_probs).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * board_tensor.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
