"""
ADI training for the 2x2 cube.

Uses a target network for stable bootstrapping (same approach as the
official DeepCubeA code). Generates a big batch of scrambled states,
computes targets using the frozen target network, trains for a few epochs,
then conditionally copies weights to the target if loss is low enough.

Usage:
    python train.py --num_updates 200 --states_per_update 50000
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_targets(states, target_net, device, all_zeros=False):
    """
    Compute ADI targets using the target network.
    target = 0 if solved, else 1 + min(V_target(neighbors))
    """
    N = states.shape[0]

    if all_zeros:
        # first update -- target net is random, just use 0/1
        solved_mask = batch_is_solved(states)
        return np.where(solved_mask, 0.0, 1.0).astype(np.float32)

    all_neighbors = batch_get_all_neighbors(states)
    flat_neighbors = all_neighbors.reshape(N * NUM_MOVES, 24)
    flat_tensor = torch.from_numpy(flat_neighbors).to(device)

    # chunk to avoid OOM
    chunk_size = 10000
    flat_values = []
    with torch.no_grad():
        for i in range(0, flat_tensor.shape[0], chunk_size):
            chunk = flat_tensor[i : i + chunk_size]
            vals = target_net(chunk)
            flat_values.append(vals.cpu())
    flat_values = torch.cat(flat_values).numpy().squeeze(-1)

    # clip negative values (same as official code)
    flat_values = np.maximum(flat_values, 0.0)

    neighbor_values = flat_values.reshape(N, NUM_MOVES)
    min_neighbor = neighbor_values.min(axis=1)

    solved_mask = batch_is_solved(states)
    targets = np.where(solved_mask, 0.0, 1.0 + min_neighbor).astype(np.float32)
    return targets


def train_on_dataset(nnet, states, targets, device, batch_size, num_epochs, lr, lr_d, train_itr):
    """Train on a fixed dataset for some epochs. Returns (last_loss, train_itr)."""
    N = states.shape[0]
    optimizer = optim.Adam(nnet.parameters(), lr=lr)
    criterion = nn.MSELoss()

    nnet.train()
    last_loss = float("inf")

    for epoch in range(num_epochs):
        perm = np.random.permutation(N)
        states_shuffled = states[perm]
        targets_shuffled = targets[perm]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if end - start < 2:
                continue

            # per-iteration lr decay
            lr_itr = lr * (lr_d ** train_itr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_itr

            x = torch.from_numpy(states_shuffled[start:end]).to(device)
            y = torch.from_numpy(targets_shuffled[start:end]).unsqueeze(1).to(device)

            pred = nnet(x)
            loss = criterion(pred.squeeze(-1), y.squeeze(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            train_itr += 1

    return last_loss, train_itr


def train(args):
    device = get_device()
    print(f"Training on: {device}")
    print(f"Target network updates when loss < {args.loss_thresh}")

    nnet = ResnetModel().to(device)
    target_net = copy.deepcopy(nnet)
    target_net.eval()

    total_params = sum(p.numel() for p in nnet.parameters())
    print(f"Model parameters: {total_params:,}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    update_num = 0
    train_itr = 0
    all_zeros = True

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

        print(f"\n[Update {update + 1}/{args.num_updates}]")
        print(f"  Generating {args.states_per_update:,} states (depth 1-{args.back_max})...")

        states, depths = batch_scramble(args.states_per_update, args.back_max)

        target_net.eval()
        targets = generate_targets(states, target_net, device, all_zeros=all_zeros)

        print(f"  Targets: mean={targets.mean():.2f}, min={targets.min():.2f}, max={targets.max():.2f}")

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

        if last_loss < args.loss_thresh:
            print(f"  ✓ Loss {last_loss:.4f} < threshold {args.loss_thresh} → updating target network")
            target_net = copy.deepcopy(nnet)
            target_net.eval()
            all_zeros = False
            update_num = update + 1

        torch.save({
            "model_state_dict": nnet.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "update_num": update + 1,
            "train_itr": train_itr,
        }, ckpt_path)
        print(f"  → Checkpoint saved")

    print(f"\nTraining complete after {args.num_updates} updates, {train_itr} training iterations.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_updates", type=int, default=200)
    parser.add_argument("--states_per_update", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=1_000)
    parser.add_argument("--epochs_per_update", type=int, default=1)
    parser.add_argument("--back_max", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_d", type=float, default=0.9999993)
    parser.add_argument("--loss_thresh", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
