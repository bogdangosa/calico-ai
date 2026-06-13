import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineStateValueNetwork(nn.Module):
    def __init__(self, input_channels: int = 13, board_size: int = 7):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv_act = nn.LeakyReLU(0.01)

        self.flattened_size = 32 * board_size * board_size
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc1_act = nn.LeakyReLU(0.01)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_act(self.conv(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc1_act(self.fc1(x))
        value = self.fc2(x)
        return value

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()