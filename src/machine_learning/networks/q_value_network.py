import torch
import torch.nn as nn
import torch.nn.functional as F
from src.machine_learning.networks.dual_head_res_net import ResidualBlock

class QValueNetwork(nn.Module):
    """
    A convolutional neural network that estimates the value (Q-value) of a board state.
    Designed to work with the CalicoEncoder output.
    """
    def __init__(self, input_channels: int = 13, board_size: int = 7, num_blocks: int = 2, hidden_channels: int = 64):
        super().__init__()
        self.board_size = board_size

        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(board_size * board_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, channels, board_size, board_size)
        Returns:
            A tensor of shape (batch_size, 1) representing the estimated value.
        """
        out = F.leaky_relu(self.bn_in(self.conv_in(x)), negative_slope=0.01)

        for block in self.res_blocks:
            out = block(out)

        out = F.relu(self.value_bn(self.value_conv(out)))
        out = out.reshape(out.size(0), -1)

        out = F.leaky_relu(self.fc1(out), negative_slope=0.01)
        value = self.fc2(out)
        
        return value

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()
