import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(  7, 64, 3, padding=1)
        self.conv1 = nn.Conv2d( 64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d( 64, 64, 3, padding=1)
        self.flat = nn.Flatten()
        self.policy_head = nn.Linear(64*8*8, 4096)  # For move distribution (64x64)
        self.value_head = nn.Linear(64*8*8, 1)      # For scalar value

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = x + F.relu(self.conv1(x))
        x = x + F.relu(self.conv2(x))
        x = self.flat(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value

    def predict(self, x, legal_mask):
        policy_logits, value = self.forward(x)
        # Set illegal logits to -1e10 before softmax
        masked_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e10)
        probs = F.softmax(masked_logits, dim=1)
        return probs, value  # (batch, 4096), (batch,)


def export_model(model):
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 7, 8, 8)

    # Export to TorchScript
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("model.pt")
    print("Model exported to model.pt")


if __name__ == "__main__":
    model = ChessModel()
    export_model(model)
