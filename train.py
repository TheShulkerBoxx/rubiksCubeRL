"""
Autodidactic Iteration (ADI) training loop for the 2x2 Rubik's Cube.

Implements the core training algorithm from the DeepCubeA paper:
  1. Generate scrambled states by working backwards from solved
  2. For each state, compute target = 1 + min(value(neighbors))
  3. Train the value network with MSE loss on these targets
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cube_env import (
    SOLVED_STATE, NUM_MOVES, ONE_HOT_DIM,
    apply_move, is_solved, scramble,
    batch_state_to_onehot, state_to_onehot,
)
from model import ValueNetwork


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_training_data(batch_size, max_depth):
    states = np.zeros((batch_size, 24), dtype=np.int8)
    depths = np.zeros(batch_size, dtype=np.int32)
    for i in range(batch_size):
        depth = np.random.randint(1, max_depth + 1)
        s, _ = scramble(depth)
        states[i] = s
        depths[i] = depth
    return states, depths


def compute_targets(states, network, device):
    N = states.shape[0]
    targets = np.zeros(N, dtype=np.float32)

    all_neighbors = np.zeros((N, NUM_MOVES, 24), dtype=np.int8)
    for m in range(NUM_MOVES):
        for i in range(N):
            all_neighbors[i, m] = apply_move(states[i], m)

    flat_neighbors = all_neighbors.reshape(N * NUM_MOVES, 24)
    flat_onehot = batch_state_to_onehot(flat_neighbors)
    flat_tensor = torch.from_numpy(flat_onehot).to(device)

    with torch.no_grad():
        flat_values = network(flat_tensor).cpu().numpy().squeeze(-1)

    neighbor_values = flat_values.reshape(N, NUM_MOVES)
    min_neighbor = neighbor_values.min(axis=1)

    for i in range(N):
        if is_solved(states[i]):
            targets[i] = 0.0
        else:
            targets[i] = 1.0 + min_neighbor[i]

    return targets


def train(args):
    device = get_device()
    print(f"Training on: {device}")

    network = ValueNetwork().to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    loss_fn = nn.HuberLoss(delta=1.0)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_iter = 0
    if args.resume and os.path.exists(os.path.join(args.checkpoint_dir, "latest.pt")):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "latest.pt"), weights_only=False)
        network.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iteration"]
        print(f"Resumed from iteration {start_iter}")

    log_interval = max(1, args.num_iterations // 100)
    running_loss = 0.0
    start_time = time.time()

    for iteration in range(start_iter, args.num_iterations):
        states, depths = generate_training_data(args.batch_size, args.max_scramble_depth)

        network.eval()
        targets = compute_targets(states, network, device)

        network.train()
        onehot = batch_state_to_onehot(states)
        x = torch.from_numpy(onehot).to(device)
        y = torch.from_numpy(targets).unsqueeze(1).to(device)

        pred = network(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (iteration + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - start_time
            iters_per_sec = (iteration + 1 - start_iter) / elapsed
            print(
                f"[Iter {iteration + 1:>7d}/{args.num_iterations}] "
                f"loss={avg_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.1e}  "
                f"speed={iters_per_sec:.1f} it/s"
            )
            running_loss = 0.0

        if (iteration + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
            torch.save({
                "iteration": iteration + 1,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

        scheduler.step()

    final_path = os.path.join(args.checkpoint_dir, "latest.pt")
    torch.save({
        "iteration": args.num_iterations,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_scramble_depth", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step", type=int, default=50000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
