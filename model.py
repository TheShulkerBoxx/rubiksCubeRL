"""
ResNet value network for the 2x2 Rubik's Cube solver.

Architecture matches the official DeepCubeA paper (Agostinelli et al., 2019):
  - One-hot encoding done inside the network via F.one_hot (GPU-native)
  - Two fully connected layers to project into residual block dimension
  - N residual blocks (FC → BN → ReLU → FC → BN + skip → ReLU)
  - Linear output head → scalar cost-to-go estimate

Input:  raw integer states (batch, 24) with values 0–5
Output: scalar value estimate (cost-to-go)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetModel(nn.Module):
    """
    ResNet value network matching the DeepCubeA architecture.

    The official DeepCubeA paper uses:
      - 3x3 cube: ResnetModel(54, 6, 5000, 1000, 4 blocks, 1 output)
      - We scale for 2x2:  ResnetModel(24, 6, 2048, 512, 4 blocks, 1 output)
    """

    def __init__(
        self,
        state_dim: int = 24,
        one_hot_depth: int = 6,
        h1_dim: int = 2048,
        resnet_dim: int = 512,
        num_resnet_blocks: int = 4,
        out_dim: int = 1,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.one_hot_depth = one_hot_depth
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm

        input_dim = state_dim * one_hot_depth  # 24 * 6 = 144

        # First two hidden layers (project into resnet dimension)
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.bn1 = nn.BatchNorm1d(h1_dim) if batch_norm else nn.Identity()

        self.fc2 = nn.Linear(h1_dim, resnet_dim)
        self.bn2 = nn.BatchNorm1d(resnet_dim) if batch_norm else nn.Identity()

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_resnet_blocks):
            block = nn.ModuleList([
                nn.Linear(resnet_dim, resnet_dim),
                nn.BatchNorm1d(resnet_dim) if batch_norm else nn.Identity(),
                nn.Linear(resnet_dim, resnet_dim),
                nn.BatchNorm1d(resnet_dim) if batch_norm else nn.Identity(),
            ])
            self.blocks.append(block)

        # Output head
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 24) integer tensor with values 0–5,
               OR (batch, 144) pre-encoded one-hot float tensor.
        Returns:
            (batch, 1) value estimates.
        """
        # One-hot encode on GPU if input is integer states
        if x.dtype in (torch.long, torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        # else: assume already float one-hot encoded

        # First two FC layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.blocks:
            residual = x
            x = block[0](x)  # FC
            x = block[1](x)  # BN
            x = F.relu(x)
            x = block[2](x)  # FC
            x = block[3](x)  # BN
            x = F.relu(x + residual)

        # Output
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    net = ResnetModel()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"ResnetModel: {total_params:,} parameters")
    print(f"  h1_dim=2048, resnet_dim=512, blocks=4")

    # Test with raw integer input (GPU-native one-hot)
    x_int = torch.randint(0, 6, (32, 24))
    v = net(x_int)
    assert v.shape == (32, 1), f"Expected (32, 1), got {v.shape}"
    print(f"Forward pass (int input) OK: {x_int.shape} → {v.shape}")

    # Test with pre-encoded float input
    x_float = torch.randn(32, 144)
    v2 = net(x_float)
    assert v2.shape == (32, 1)
    print(f"Forward pass (float input) OK: {x_float.shape} → {v2.shape}")

    # Single-sample eval mode
    net.eval()
    x1 = torch.randint(0, 6, (1, 24))
    v1 = net(x1)
    assert v1.shape == (1, 1)
    print("Single-sample eval OK ✓")
