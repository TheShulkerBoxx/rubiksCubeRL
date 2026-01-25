"""
Value network for the 2x2 Rubik's Cube solver.

Architecture: fully-connected layers with batch normalization and ELU.
Input:  144-dim one-hot encoded cube state
Output: scalar value estimate (cost-to-go)
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, input_dim=144, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4096, 2048, 512]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ELU(),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.value_head(features)


if __name__ == "__main__":
    net = ValueNetwork()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"ValueNetwork: {total_params:,} parameters")

    x = torch.randn(32, 144)
    v = net(x)
    assert v.shape == (32, 1)
    print(f"Forward pass OK: {x.shape} -> {v.shape}")
