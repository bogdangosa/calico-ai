import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        # Second block
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        out += identity
        out = F.leaky_relu(out, negative_slope=0.01)

        return out


class DualHeadResNet2(nn.Module):
    def __init__(self, input_channels: int, board_size: int, num_actions: int, 
                 flat_features_size: int = 0, num_blocks: int = 3,
                 hidden_channels: int = 64):
        super().__init__()
        self.board_size = board_size
        self.flat_features_size = flat_features_size
        
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_linear = nn.Linear(2 * board_size * board_size + flat_features_size, num_actions)

        # Value Head
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_linear1 = nn.Linear(board_size * board_size + flat_features_size, 64)
        self.value_linear2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, flat_features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head logic
        pol = F.relu(self.policy_bn(self.policy_conv(out)))
        pol = pol.view(pol.size(0), -1)
        if self.flat_features_size > 0 and flat_features is not None:
            pol = torch.cat([pol, flat_features], dim=1)
        policy_logits = self.policy_linear(pol)
        policy_probs = F.softmax(policy_logits, dim=1)

        # Value head logic
        val = F.relu(self.value_bn(self.value_conv(out)))
        val = val.view(val.size(0), -1)
        if self.flat_features_size > 0 and flat_features is not None:
            val = torch.cat([val, flat_features], dim=1)
        val = F.relu(self.value_linear1(val))
        value = self.value_linear2(val)

        return policy_probs, value
