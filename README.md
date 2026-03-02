# DeepCubeA: Solving the 2×2 Rubik's Cube with Deep Reinforcement Learning

A deep reinforcement learning solver for the 2×2 Pocket Cube, implementing the Autodidactic Iteration algorithm from the [DeepCubeA paper](https://www.nature.com/articles/s42256-019-0070-z) (Agostinelli et al., 2019).

A ResNet value network learns a cost-to-go function entirely from self-play — working backwards from the solved state using a target network for stable bootstrapping — then guides A\* search to find solutions.

## The 2×2 Rubik's Cube as a Group

The 2×2 Pocket Cube is a mathematically elegant object.  Its set of reachable configurations forms a [group](https://en.wikipedia.org/wiki/Group_(mathematics)) under composition of moves:

| Property | Value |
|---|---|
| Reachable states | **3,674,160** |
| Group structure | $(\\mathbb{Z}_3^7) \\rtimes S_7$ (orientations ⋊ permutations) |
| God's number (QTM) | **14** (every state solvable in ≤14 quarter turns) |
| God's number (HTM) | **11** |

**Solving the cube means finding a path through this group back to the identity element** (the solved state).  Each face turn is a group element, and a solution is a word in the generators $\\{R, R', U, U', F, F'\\}$ that maps the scrambled state to the identity.

## Approach: DeepCubeA & Autodidactic Iteration

### The Problem

Classical RL struggles with the Rubik's Cube because:
- **Sparse reward**: only one state out of millions is "solved"
- **No natural curriculum**: random exploration almost never reaches the goal
- **Huge state space**: even the 2×2 has 3.67M states

### Autodidactic Iteration (ADI)

The key insight is to **train in reverse** — start from the solved state, scramble, and learn to undo the scramble:

```
for each update cycle:
    1. Generate N scrambled states from the SOLVED state
    2. Use the FROZEN TARGET NETWORK to compute targets:
         For each neighbor s' of s:  compute V_target(s')
         Target: y(s) = 1 + min_a V_target(apply(s, a))
         (solved state gets target 0)
    3. Train the CURRENT network on this fixed dataset with MSE loss
    4. If loss < threshold: copy current → target network
```

The **target network** is the key to stability: targets don't shift during training within an update cycle, preventing the bootstrapping divergence that naive ADI suffers from.

The value function $V(s)$ learns to estimate the **cost-to-go**: the minimum number of moves needed to solve state $s$.

### Network Architecture (matches the paper)

```
Input (24 ints) → F.one_hot → FC(144→2048) → BN → ReLU
    → FC(2048→512) → BN → ReLU
    → [ResBlock(512→512)]×4       # FC→BN→ReLU→FC→BN + skip → ReLU
    → FC(512→1)                   # cost-to-go estimate
```

### Guided Search

Once trained, the value network serves as a heuristic for **weighted A\* search**:

$$f(s) = g(s) + \\lambda \\cdot h(s)$$

where $g(s)$ is the path cost so far, $h(s) = V(s)$ is the neural network estimate, and $\\lambda$ is a weight that trades optimality for speed.

## Project Structure

```
rubiksCubeRL/
├── cube_env.py       # 2×2 cube environment (vectorized, batch operations)
├── model.py          # ResNet value network (matches DeepCubeA architecture)
├── train.py          # ADI training with target network + batch-epoch pattern
├── solve.py          # Weighted A* search using the trained network
├── requirements.txt  # Python dependencies
├── LICENSE           # GPLv3
└── README.md
```

## How to Run

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
# Quick test (~2 minutes)
python train.py --num_updates 10 --states_per_update 10000 --back_max 15

# Full training (may take 30min–2hr depending on hardware)
python train.py --num_updates 200 --states_per_update 50000 --back_max 30

# Resume training
python train.py --resume
```

Checkpoints are saved to `checkpoints/latest.pt`.

### Solve

```bash
# Solve 10 random scrambles of depth 7
python solve.py --model_path checkpoints/latest.pt --scramble_depth 7

# Harder scrambles, larger search budget
python solve.py --model_path checkpoints/latest.pt --scramble_depth 14 --max_nodes 50000

# Weighted A* for faster (but possibly suboptimal) solutions
python solve.py --model_path checkpoints/latest.pt --scramble_depth 14 --weight 5.0
```

## Results

After training for 10k–50k iterations:
- **Near-100% solve rate** on scrambles up to depth ~10
- **Solution quality improves** with more training iterations and higher search budgets
- **Shallow scrambles** (depth 1–5) are solved almost instantly

*(Exact results depend on training duration and hardware.)*

## Future Work

- **3×3 Rubik's Cube**: The full cube has **43,252,003,274,489,856,000** (43 quintillion) reachable states.  The same ADI approach applies but requires significantly more compute and a larger network.  The original DeepCubeA paper trained on ~8 billion states over 44 hours.
- **Policy head**: Add a policy output to the network for more directed search
- **Batch-weighted A\* (BWAS)**: Expand multiple nodes per iteration for GPU-efficient search
- **Curriculum learning**: Gradually increase scramble depth during training
- **Symmetry exploitation**: Use the cube's symmetry group to reduce the effective state space

## References

1. Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). **Solving the Rubik's Cube with Deep Reinforcement Learning and Search.** *Nature Machine Intelligence, 1*, 356–363. [doi:10.1038/s42256-019-0070-z](https://doi.org/10.1038/s42256-019-0070-z)

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
