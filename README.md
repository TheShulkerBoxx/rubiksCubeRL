# DeepCubeA 2x2 Solver

Solving the 2x2 Rubik's cube with deep RL, based on the DeepCubeA paper (Agostinelli et al., 2019).

Trains a neural network to estimate how many moves any state is from solved, then uses that as a heuristic for A* search.

## How it works

The 2x2 cube has about 3.67 million reachable states and God's number is 14 (quarter turn metric).

**Training (Autodidactic Iteration):**

The idea is to train backwards from the solved state. You scramble a bunch of cubes, then for each scrambled state, look at all its neighbors (states reachable in one move) and ask the network "how far are these from solved?". The target is `1 + min(neighbor values)`. Solved states get target 0.

To keep things stable, there's a frozen "target network" that generates the targets. The actual network being trained is separate. You only copy the trained weights into the target network once the loss drops below a threshold. This prevents the feedback loop where bad predictions make bad targets which make worse predictions (which absolutely does happen without it -- ask me how I know).

**Solving:**

Once trained, the value network is used as a heuristic for weighted A* search. Works pretty well for scrambles up to depth ~10-12 with moderate training.

## Architecture

ResNet with residual blocks, based on the architecture from the actual DeepCubeA repo:

```
input (24 sticker values) -> one-hot (144d) -> FC 2048 -> FC 512 -> 4 res blocks -> output (1 value)
```

~3.5M parameters.

## Usage

```
pip install -r requirements.txt

# train (takes a few minutes for basic results, longer for better ones)
python train.py --num_updates 200 --states_per_update 50000 --back_max 30

# resume if you stopped it
python train.py --resume

# solve some scrambles
python solve.py --model_path checkpoints/latest.pt --scramble_depth 10

# harder scrambles, bigger search
python solve.py --model_path checkpoints/latest.pt --scramble_depth 14 --max_nodes 50000
```

## Files

- `cube_env.py` - 2x2 cube environment, moves, state representation
- `model.py` - ResNet value network
- `train.py` - training loop with target network
- `solve.py` - A* solver

## References

Agostinelli et al., "Solving the Rubik's Cube with Deep Reinforcement Learning and Search", Nature Machine Intelligence, 2019.

## License

GPLv3
