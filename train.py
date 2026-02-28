"""
Autodidactic Iteration (ADI) training for the 2x2 Rubik's Cube.
Matches the official DeepCubeA training approach.

Key features (from the paper / official repo):
  1. Target network — frozen copy for stable target generation
  2. Batch-epoch training — generate large dataset, train multiple epochs
  3. Target network updated only when loss drops below threshold
  4. Exponential LR decay per iteration
  5. Raw integer states sent to GPU, one-hot done in-network

Usage:
    python train.py --num_updates 100 --states_per_update 50000
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cube_env import (
    SOLVED_STATE,
    NUM_MOVES,
    MOVE_PERMS,
    batch_scramble,
    batch_get_all_neighbors,
    batch_is_solved,
)
from model import ResnetModel


def get_device() -> torch.device:
    """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_targets(
    states: np.ndarray,
    target_net: ResnetModel,
    device: torch.device,
    all_zeros: bool = False,
) -> np.ndarray:
    """
    Compute ADI targets for a batch of states using the TARGET network.

    target(s) = 0                           if s is solved
    target(s) = 1 + min_a V_target(s')      otherwise

    When all_zeros=True (first update, untrained target), targets are set
    to 0 for solved states and 1 for all others.
    """
    N = states.shape[0]

    if all_zeros:
        solved_mask = batch_is_solved(states)
        return np.where(solved_mask, 0.0, 1.0).astype(np.float32)

    # Get all neighbors: (N, NUM_MOVES, 24)
    all_neighbors = batch_get_all_neighbors(states)
    flat_neighbors = all_neighbors.reshape(N * NUM_MOVES, 24)

    # Batch evaluate on GPU using target network
    flat_tensor = torch.from_numpy(flat_neighbors).to(device)

    chunk_size = 10000
    flat_values = []
    with torch.no_grad():
        for i in range(0, flat_tensor.shape[0], chunk_size):
            chunk = flat_tensor[i : i + chunk_size]
            vals = target_net(chunk)
            flat_values.append(vals.cpu())
    flat_values = torch.cat(flat_values).numpy().squeeze(-1)

    # Clip values to >= 0 (matching official code: clip_zero=True)
    flat_values = np.maximum(flat_values, 0.0)

    neighbor_values = flat_values.reshape(N, NUM_MOVES)
    min_neighbor = neighbor_values.min(axis=1)

    solved_mask = batch_is_solved(states)
    targets = np.where(solved_mask, 0.0, 1.0 + min_neighbor).astype(np.float32)

    return targets


def train_on_dataset(
    nnet: ResnetModel,
    states: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_epochs: int,
    lr: float,
    lr_d: float,
    train_itr: int,
) -> tuple[float, int]:
    """
    Train the network on a fixed dataset for num_epochs epochs.
    Returns (last_loss, updated_train_itr).
    """
    N = states.shape[0]
    optimizer = optim.Adam(nnet.parameters(), lr=lr)
    criterion = nn.MSELoss()

    nnet.train()
    last_loss = float("inf")

    for epoch in range(num_epochs):
        # Shuffle dataset each epoch
        perm = np.random.permutation(N)
        states_shuffled = states[perm]
        targets_shuffled = targets[perm]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if end - start < 2:  # skip tiny batches (BatchNorm needs ≥2)
                continue

            # Exponential LR decay per iteration
            lr_itr = lr * (lr_d ** train_itr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_itr

            # Send raw int states to GPU (one-hot inside network)
            x = torch.from_numpy(states_shuffled[start:end]).to(device)
            y = torch.from_numpy(targets_shuffled[start:end]).unsqueeze(1).to(device)

            pred = nnet(x)
            loss = criterion(pred.squeeze(-1), y.squeeze(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            epoch_loss += last_loss
            num_batches += 1
            train_itr += 1

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches

    return last_loss, train_itr


def train(args):
    """Main training loop with target network pattern."""
    device = get_device()
    print(f"Training on: {device}")
    print(f"Target network updates when loss < {args.loss_thresh}")

    # Initialize current and target networks
    nnet = ResnetModel().to(device)
    target_net = copy.deepcopy(nnet)
    target_net.eval()

    total_params = sum(p.numel() for p in nnet.parameters())
    print(f"Model parameters: {total_params:,}")

    # Checkpointing
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    update_num = 0
    train_itr = 0
    all_zeros = True  # First update: target is untrained

    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        nnet.load_state_dict(ckpt["model_state_dict"])
        target_net.load_state_dict(ckpt["target_state_dict"])
        update_num = ckpt.get("update_num", 0)
        train_itr = ckpt.get("train_itr", 0)
        all_zeros = False
        print(f"Resumed: update_num={update_num}, train_itr={train_itr}")

    start_time = time.time()

    for update in range(update_num, args.num_updates):
        update_start = time.time()

        # ── 1. Generate training data using TARGET network ────────────
        print(f"\n[Update {update + 1}/{args.num_updates}]")
        print(f"  Generating {args.states_per_update:,} states (depth 1–{args.back_max})...")

        states, depths = batch_scramble(args.states_per_update, args.back_max)

        target_net.eval()
        targets = generate_targets(states, target_net, device, all_zeros=all_zeros)

        print(f"  Targets: mean={targets.mean():.2f}, min={targets.min():.2f}, max={targets.max():.2f}")

        # ── 2. Train current network on this fixed dataset ────────────
        num_train_itrs = args.epochs_per_update * int(np.ceil(len(targets) / args.batch_size))
        print(f"  Training for {args.epochs_per_update} epoch(s) ({num_train_itrs} iterations)...")

        last_loss, train_itr = train_on_dataset(
            nnet, states, targets, device,
            batch_size=args.batch_size,
            num_epochs=args.epochs_per_update,
            lr=args.lr, lr_d=args.lr_d,
            train_itr=train_itr,
        )

        elapsed = time.time() - update_start
        total_elapsed = time.time() - start_time
        print(f"  Loss: {last_loss:.4f} | LR: {args.lr * (args.lr_d ** train_itr):.2e} | "
              f"Update time: {elapsed:.1f}s | Total: {total_elapsed:.0f}s")

        # ── 3. Conditionally update target network ───────────────────
        if last_loss < args.loss_thresh:
            print(f"  ✓ Loss {last_loss:.4f} < threshold {args.loss_thresh} → updating target network")
            target_net = copy.deepcopy(nnet)
            target_net.eval()
            all_zeros = False
            update_num = update + 1

        # ── 4. Save checkpoint ────────────────────────────────────────
        torch.save(
            {
                "model_state_dict": nnet.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "update_num": update + 1,
                "train_itr": train_itr,
            },
            ckpt_path,
        )
        print(f"  → Checkpoint saved")

    print(f"\nTraining complete after {args.num_updates} updates, {train_itr} training iterations.")


def main():
    parser = argparse.ArgumentParser(description="ADI Training for 2x2 Rubik's Cube (DeepCubeA)")
    parser.add_argument("--num_updates", type=int, default=200,
                        help="Number of target network update cycles")
    parser.add_argument("--states_per_update", type=int, default=50_000,
                        help="States to generate per update cycle")
    parser.add_argument("--batch_size", type=int, default=1_000,
                        help="Training batch size")
    parser.add_argument("--epochs_per_update", type=int, default=1,
                        help="Training epochs per update cycle")
    parser.add_argument("--back_max", type=int, default=30,
                        help="Maximum scramble depth")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--lr_d", type=float, default=0.9999993,
                        help="LR decay per iteration: lr * (lr_d ^ itr)")
    parser.add_argument("--loss_thresh", type=float, default=0.1,
                        help="Update target network when loss falls below this")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
