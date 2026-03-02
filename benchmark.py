"""
Benchmark script -- tests the solver at various scramble depths
and generates performance graphs.

Usage:
    python benchmark.py --model_path checkpoints/latest.pt
"""

import argparse
import time
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

from cube_env import MOVE_NAMES, scramble, is_solved
from model import ResnetModel
from solve import get_device, weighted_astar


def benchmark_depth(network, device, depth, num_trials=50, max_nodes=50000, weight=1.0):
    """Run num_trials solves at a given scramble depth."""
    results = {
        'solved': 0,
        'total': num_trials,
        'move_lengths': [],
        'nodes': [],
        'times': [],
    }

    for _ in range(num_trials):
        state, moves = scramble(depth)

        result = weighted_astar(
            state, network, device,
            weight=weight,
            max_nodes=max_nodes,
            verbose=False,
        )

        if result['solved']:
            results['solved'] += 1
            results['move_lengths'].append(len(result['solution']))
            results['nodes'].append(result['nodes_expanded'])
            results['times'].append(result['time'])
        else:
            results['nodes'].append(result['nodes_expanded'])
            results['times'].append(result['time'])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/latest.pt')
    parser.add_argument('--max_depth', type=int, default=14)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--max_nodes', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    network = ResnetModel()
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    network.load_state_dict(ckpt['model_state_dict'])
    network = network.to(device)
    network.eval()
    print(f"Loaded model from {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    depths = list(range(1, args.max_depth + 1))
    solve_rates = []
    avg_moves = []
    avg_nodes = []
    avg_times = []

    print(f"\nBenchmarking depths 1-{args.max_depth}, {args.trials} trials each\n")
    print(f"{'Depth':>5}  {'Solved':>8}  {'Rate':>6}  {'Avg Moves':>10}  {'Avg Nodes':>10}  {'Avg Time':>9}")
    print("-" * 58)

    for d in depths:
        res = benchmark_depth(network, device, d, num_trials=args.trials, max_nodes=args.max_nodes)

        rate = res['solved'] / res['total'] * 100
        solve_rates.append(rate)

        if res['move_lengths']:
            am = np.mean(res['move_lengths'])
            avg_moves.append(am)
        else:
            avg_moves.append(0)

        an = np.mean(res['nodes'])
        avg_nodes.append(an)

        at = np.mean(res['times'])
        avg_times.append(at)

        print(f"{d:>5}  {res['solved']:>4}/{res['total']:<3}  {rate:>5.1f}%  "
              f"{avg_moves[-1]:>10.1f}  {an:>10.1f}  {at:>8.3f}s")

    # save raw data
    np.savez(
        os.path.join(args.output_dir, 'benchmark_data.npz'),
        depths=depths,
        solve_rates=solve_rates,
        avg_moves=avg_moves,
        avg_nodes=avg_nodes,
        avg_times=avg_times,
    )

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('DeepCubeA 2x2 Solver Performance', fontsize=14, fontweight='bold')

    # solve rate
    ax = axes[0, 0]
    ax.bar(depths, solve_rates, color='#4CAF50', alpha=0.85, edgecolor='#388E3C')
    ax.set_xlabel('Scramble Depth')
    ax.set_ylabel('Solve Rate (%)')
    ax.set_title('Solve Rate by Depth')
    ax.set_ylim(0, 105)
    ax.set_xticks(depths)
    for i, v in enumerate(solve_rates):
        if v < 100:
            ax.text(depths[i], v + 2, f'{v:.0f}%', ha='center', fontsize=8)

    # avg solution length
    ax = axes[0, 1]
    ax.plot(depths, avg_moves, 'o-', color='#2196F3', linewidth=2, markersize=5)
    ax.set_xlabel('Scramble Depth')
    ax.set_ylabel('Avg Solution Length (moves)')
    ax.set_title('Solution Length by Depth')
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)

    # avg nodes expanded
    ax = axes[1, 0]
    ax.plot(depths, avg_nodes, 's-', color='#FF9800', linewidth=2, markersize=5)
    ax.set_xlabel('Scramble Depth')
    ax.set_ylabel('Avg Nodes Expanded')
    ax.set_title('Search Effort by Depth')
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)

    # avg solve time
    ax = axes[1, 1]
    ax.plot(depths, avg_times, '^-', color='#9C27B0', linewidth=2, markersize=5)
    ax.set_xlabel('Scramble Depth')
    ax.set_ylabel('Avg Time (seconds)')
    ax.set_title('Solve Time by Depth')
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'benchmark.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved graph to {fig_path}")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
