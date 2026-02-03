"""
Solver for the 2x2 Rubik's Cube using weighted A* search
guided by a trained value network.
"""

import argparse
import heapq
import time

import numpy as np
import torch

from cube_env import (
    SOLVED_STATE, NUM_MOVES, MOVE_NAMES,
    apply_move, is_solved, scramble,
    state_to_onehot, batch_state_to_onehot, state_to_bytes,
)
from model import ValueNetwork


def evaluate_states(states, network, device):
    if not states:
        return np.array([])
    batch = np.stack(states)
    onehot = batch_state_to_onehot(batch)
    tensor = torch.from_numpy(onehot).to(device)
    with torch.no_grad():
        values = network(tensor).cpu().numpy().squeeze(-1)
    return values


def weighted_astar(start_state, network, device, weight=1.0, max_nodes=10000, verbose=True):
    network.eval()

    if is_solved(start_state):
        return {"solved": True, "solution": [], "solution_moves": [],
                "nodes_expanded": 0, "time": 0.0}

    h0 = evaluate_states([start_state], network, device)[0]
    counter = 0
    start_bytes = state_to_bytes(start_state)
    open_set = [(weight * h0, counter, start_bytes, 0, [])]
    counter += 1
    g_scores = {start_bytes: 0}
    state_cache = {start_bytes: start_state.copy()}
    nodes_expanded = 0
    start_time = time.time()

    while open_set and nodes_expanded < max_nodes:
        f_score, _, state_bytes, g, move_history = heapq.heappop(open_set)
        if state_bytes in g_scores and g > g_scores[state_bytes]:
            continue

        state = state_cache[state_bytes]
        nodes_expanded += 1

        neighbor_states = []
        neighbor_moves = []
        neighbor_g = g + 1

        for m in range(NUM_MOVES):
            child = apply_move(state, m)
            child_bytes = state_to_bytes(child)

            if is_solved(child):
                elapsed = time.time() - start_time
                solution_moves = move_history + [m]
                return {"solved": True,
                        "solution": [MOVE_NAMES[mi] for mi in solution_moves],
                        "solution_moves": solution_moves,
                        "nodes_expanded": nodes_expanded, "time": elapsed}

            if child_bytes in g_scores and neighbor_g >= g_scores[child_bytes]:
                continue
            g_scores[child_bytes] = neighbor_g
            state_cache[child_bytes] = child
            neighbor_states.append(child)
            neighbor_moves.append(m)

        if neighbor_states:
            h_values = evaluate_states(neighbor_states, network, device)
            for child, m, h in zip(neighbor_states, neighbor_moves, h_values):
                child_bytes = state_to_bytes(child)
                f = neighbor_g + weight * max(h, 0)
                heapq.heappush(open_set, (f, counter, child_bytes, neighbor_g, move_history + [m]))
                counter += 1

    elapsed = time.time() - start_time
    return {"solved": False, "solution": [], "solution_moves": [],
            "nodes_expanded": nodes_expanded, "time": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scramble_depth", type=int, default=10)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--max_nodes", type=int, default=10000)
    parser.add_argument("--num_solves", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    network = ValueNetwork()
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    network.load_state_dict(ckpt["model_state_dict"])
    network = network.to(device)
    network.eval()
    print(f"Loaded model (trained for {ckpt['iteration']} iterations)")

    solved_count = 0
    total_moves = 0
    total_nodes = 0
    total_time = 0.0

    print(f"\nSolving {args.num_solves} scrambles (depth={args.scramble_depth})...\n")

    for i in range(args.num_solves):
        state, scramble_moves = scramble(args.scramble_depth)
        scramble_str = " ".join(MOVE_NAMES[m] for m in scramble_moves)
        result = weighted_astar(state, network, device, weight=args.weight,
                                max_nodes=args.max_nodes, verbose=False)

        if result["solved"]:
            solved_count += 1
            sol_len = len(result["solution"])
            total_moves += sol_len
            total_nodes += result["nodes_expanded"]
            total_time += result["time"]
            print(f"  [{i+1:3d}] SOLVED in {sol_len:2d} moves, "
                  f"{result['nodes_expanded']:5d} nodes, {result['time']:.3f}s")
        else:
            total_nodes += result["nodes_expanded"]
            total_time += result["time"]
            print(f"  [{i+1:3d}] FAILED after {result['nodes_expanded']} nodes")

    print(f"\nResults: {solved_count}/{args.num_solves} solved")
    if solved_count > 0:
        print(f"  Avg solution length: {total_moves/solved_count:.1f}")
    print(f"  Avg nodes: {total_nodes/args.num_solves:.0f}")
    print(f"  Avg time: {total_time/args.num_solves:.3f}s")


if __name__ == "__main__":
    main()
