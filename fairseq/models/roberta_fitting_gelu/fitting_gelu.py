import torch
import torch.nn.functional as F
import torch.nn as nn


class FittingGelu(nn.Module):
    def __init__(self, hidden_features: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x.unsqueeze(-1)))
        x = self.fc2(x)
        return x.squeeze(-1)
