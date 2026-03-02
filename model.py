"""
ResNet value network for the 2x2 cube.

Based on the architecture in the DeepCubeA repo (forestagostinelli/DeepCubeA).
One-hot encoding is done inside the network via F.one_hot so we can pass
raw integer states directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetModel(nn.Module):
    """
    ResNet value network. Takes raw cube states (24 ints) and outputs
    a scalar cost-to-go estimate.

    Paper uses 5000/1000 hidden dims for 3x3 -- we use 2048/512 for 2x2.
    """

    def __init__(
        self,
        state_dim=24,
        one_hot_depth=6,
        h1_dim=2048,
        resnet_dim=512,
        num_resnet_blocks=4,
        out_dim=1,
        batch_norm=True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.one_hot_depth = one_hot_depth
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm

        input_dim = state_dim * one_hot_depth  # 144

        # project to hidden dim
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.bn1 = nn.BatchNorm1d(h1_dim) if batch_norm else nn.Identity()

        self.fc2 = nn.Linear(h1_dim, resnet_dim)
        self.bn2 = nn.BatchNorm1d(resnet_dim) if batch_norm else nn.Identity()

        # res blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_resnet_blocks):
            block = nn.ModuleList([
                nn.Linear(resnet_dim, resnet_dim),
                nn.BatchNorm1d(resnet_dim) if batch_norm else nn.Identity(),
                nn.Linear(resnet_dim, resnet_dim),
                nn.BatchNorm1d(resnet_dim) if batch_norm else nn.Identity(),
            ])
            self.blocks.append(block)

        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, x):
        # one-hot on GPU if we got integer input
        if x.dtype in (torch.long, torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        for block in self.blocks:
            residual = x
            x = F.relu(block[1](block[0](x)))
            x = block[3](block[2](x))
            x = F.relu(x + residual)

        return self.fc_out(x)


if __name__ == "__main__":
    net = ResnetModel()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"ResnetModel: {total_params:,} parameters")

    # test with int input
    x = torch.randint(0, 6, (32, 24))
    v = net(x)
    assert v.shape == (32, 1)
    print(f"Forward pass OK: {x.shape} -> {v.shape}")

    # test with float input
    x2 = torch.randn(32, 144)
    v2 = net(x2)
    assert v2.shape == (32, 1)
    print("Float input OK")

    net.eval()
    v3 = net(torch.randint(0, 6, (1, 24)))
    assert v3.shape == (1, 1)
    print("Single sample eval OK")
