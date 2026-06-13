import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineQNetwork(nn.Module):

    def __init__(self, input_channels: int = 13, board_size: int = 7, flat_features_size: int = 0, action_space_size: int = 11) -> None:
        super().__init__()

        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv_act = nn.LeakyReLU(0.01)

        self.flattened_board_size = 32 * board_size * board_size
        
        # Combined size: flattened board + flat features
        self.fc1 = nn.Linear(self.flattened_board_size + flat_features_size, 256)
        self.fc1_act = nn.LeakyReLU(0.01)

        self.fc2 = nn.Linear(256, action_space_size)

    def forward(self, board: torch.Tensor, flat_features: torch.Tensor) -> torch.Tensor:
        x = self.conv_act(self.conv(board))
        x = x.reshape(x.size(0), -1)
        
        # Concatenate board features and flat features
        combined = torch.cat([x, flat_features], dim=1)
        
        x = self.fc1_act(self.fc1(combined))
        q_values = self.fc2(x)
        return q_values

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device = None):
        if device is None:
            # Fallback to current device or CPU
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()