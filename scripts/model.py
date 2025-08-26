import torch
import torch.nn as nn


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        # Input: 7x8x8 (7 channels, 8x8 board)
        self.conv_blocks = nn.ModuleList()

        # First conv block
        self.conv_blocks.append(self._make_conv_block(7, 128))

        # 7 more conv blocks (8 total)
        for _ in range(7):
            self.conv_blocks.append(self._make_conv_block(128, 128))

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64 * 64),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1), nn.Tanh()
        )

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # x shape: (batch_size, 7, 8, 8)
        identity = None

        for i, block in enumerate(self.conv_blocks):
            if i == 0:
                x = block(x)
                identity = x
            else:
                residual = x
                x = block(x)
                x = x + residual  # Residual connection
            x = torch.relu(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


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
