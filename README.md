# DeepCubeA: Solving the 2x2 Rubik's Cube with Deep Reinforcement Learning

A deep reinforcement learning solver for the 2x2 Pocket Cube, implementing the
Autodidactic Iteration algorithm from the [DeepCubeA paper](https://www.nature.com/articles/s42256-019-0070-z)
(Agostinelli et al., 2019).

## The 2x2 Rubik's Cube as a Group

The 2x2 Pocket Cube is a mathematically elegant object. Its set of reachable
configurations forms a group under composition of moves:

| Property | Value |
|---|---|
| Reachable states | **3,674,160** |
| God's number (QTM) | **14** |
| God's number (HTM) | **11** |

Solving the cube means finding a path through this group back to the identity
element (the solved state). Each face turn is a group element, and a solution is
a word in the generators {R, R', U, U', F, F'} that maps the scrambled state to
the identity.

## Approach

Uses Autodidactic Iteration (ADI) from the DeepCubeA paper:

1. Start from the solved state, scramble with random moves
2. Target value = 1 + min(value of all neighbor states)
3. Train neural network with MSE loss
4. Use trained network as heuristic for weighted A* search

## Project Structure

```
cube_env.py       - 2x2 cube environment
model.py          - value network
train.py          - ADI training loop
solve.py          - weighted A* solver
```

## Usage

```bash
pip install -r requirements.txt

# train
python train.py --num_iterations 10000 --batch_size 512

# solve
python solve.py --model_path checkpoints/latest.pt --scramble_depth 10
```

## Future Work

- Extend to the 3x3 Rubik's Cube (43 quintillion states)
- Add policy head for more directed search
- Exploit cube symmetries to reduce state space

## References

Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). Solving the
Rubik's Cube with Deep Reinforcement Learning and Search. Nature Machine
Intelligence, 1, 356-363.

## License

GPLv3
