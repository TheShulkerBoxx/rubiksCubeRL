"""
2x2 Rubik's Cube environment.

Represents the 2x2 Pocket Cube with 24 stickers (6 faces x 4 stickers).
Uses quarter-turn metric with 6 moves: R, R', U, U', F, F'.
The DBL (Down-Back-Left) corner is fixed so only 3 faces are movable,
giving ~3.67 million reachable states.

State encoding: flat numpy array of 24 integers (0-5), one per sticker color.
One-hot encoding: 144-dim vector (24 positions x 6 colors) for neural network input.
"""

import numpy as np
from copy import deepcopy


# Face layout:
#  Each face has 4 stickers indexed as:
#       0  1
#       2  3
#
#  Faces (in order of the 24-element state array):
#    indices  0- 3 : U (Up)        color 0
#    indices  4- 7 : D (Down)      color 1
#    indices  8-11 : F (Front)     color 2
#    indices 12-15 : B (Back)      color 3
#    indices 16-19 : L (Left)      color 4
#    indices 20-23 : R (Right)     color 5

SOLVED_STATE = np.array(
    [0, 0, 0, 0,
     1, 1, 1, 1,
     2, 2, 2, 2,
     3, 3, 3, 3,
     4, 4, 4, 4,
     5, 5, 5, 5],
    dtype=np.int8,
)

NUM_STICKERS = 24
NUM_COLORS = 6
ONE_HOT_DIM = NUM_STICKERS * NUM_COLORS  # 144

# Move definitions - each move is a permutation on the 24-sticker array
# R (Right face clockwise)
_R_PERM = list(range(24))
_R_PERM[20], _R_PERM[21], _R_PERM[23], _R_PERM[22] = 22, 20, 21, 23
_R_PERM[1], _R_PERM[3] = 9, 11
_R_PERM[9], _R_PERM[11] = 5, 7
_R_PERM[5], _R_PERM[7] = 14, 12
_R_PERM[14], _R_PERM[12] = 1, 3

# U (Up face clockwise)
_U_PERM = list(range(24))
_U_PERM[0], _U_PERM[1], _U_PERM[3], _U_PERM[2] = 2, 0, 1, 3
_U_PERM[8], _U_PERM[9] = 20, 21
_U_PERM[20], _U_PERM[21] = 12, 13
_U_PERM[12], _U_PERM[13] = 16, 17
_U_PERM[16], _U_PERM[17] = 8, 9

# F (Front face clockwise)
_F_PERM = list(range(24))
_F_PERM[8], _F_PERM[9], _F_PERM[11], _F_PERM[10] = 10, 8, 9, 11
_F_PERM[2], _F_PERM[3] = 20, 22
_F_PERM[20], _F_PERM[22] = 5, 4
_F_PERM[4], _F_PERM[5] = 19, 17
_F_PERM[17], _F_PERM[19] = 2, 3


def _invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


# 0: R   1: R'   2: U   3: U'   4: F   5: F'
MOVE_PERMS = [
    np.array(_R_PERM, dtype=np.int8),
    np.array(_invert_perm(_R_PERM), dtype=np.int8),
    np.array(_U_PERM, dtype=np.int8),
    np.array(_invert_perm(_U_PERM), dtype=np.int8),
    np.array(_F_PERM, dtype=np.int8),
    np.array(_invert_perm(_F_PERM), dtype=np.int8),
]

MOVE_NAMES = ["R", "R'", "U", "U'", "F", "F'"]
NUM_MOVES = len(MOVE_PERMS)


def apply_move(state, move_idx):
    return state[MOVE_PERMS[move_idx]]


def is_solved(state):
    return np.array_equal(state, SOLVED_STATE)


def scramble(n, state=None):
    if state is None:
        state = SOLVED_STATE.copy()
    else:
        state = state.copy()
    moves = []
    for _ in range(n):
        m = np.random.randint(NUM_MOVES)
        state = apply_move(state, m)
        moves.append(m)
    return state, moves


def get_neighbors(state):
    return [(apply_move(state, m), m) for m in range(NUM_MOVES)]


def state_to_onehot(state):
    onehot = np.zeros(ONE_HOT_DIM, dtype=np.float32)
    for i, color in enumerate(state):
        onehot[i * NUM_COLORS + color] = 1.0
    return onehot


def batch_state_to_onehot(states):
    N = states.shape[0]
    onehot = np.zeros((N, ONE_HOT_DIM), dtype=np.float32)
    positions = np.arange(NUM_STICKERS)
    for i in range(N):
        onehot[i, positions * NUM_COLORS + states[i]] = 1.0
    return onehot


def state_to_bytes(state):
    return state.tobytes()


class Cube2x2:
    def __init__(self, state=None):
        self.state = state.copy() if state is not None else SOLVED_STATE.copy()

    def apply_move(self, move_idx):
        self.state = apply_move(self.state, move_idx)

    def is_solved(self):
        return is_solved(self.state)

    def get_state(self):
        return self.state.copy()

    def scramble(self, n):
        self.state, _ = scramble(n, self.state)
        return self.get_state()

    @staticmethod
    def state_to_onehot(state):
        return state_to_onehot(state)

    def __repr__(self):
        faces = ["U", "D", "F", "B", "L", "R"]
        lines = []
        for i, f in enumerate(faces):
            s = self.state[i * 4 : (i + 1) * 4]
            lines.append(f"{f}: {s}")
        return "\n".join(lines)


if __name__ == "__main__":
    c = Cube2x2()
    assert c.is_solved()

    c.apply_move(0)
    assert not c.is_solved()
    c.apply_move(1)
    assert c.is_solved()

    for move_idx in range(NUM_MOVES):
        c2 = Cube2x2()
        for _ in range(4):
            c2.apply_move(move_idx)
        assert c2.is_solved(), f"4x {MOVE_NAMES[move_idx]} should be identity"

    print("All cube_env tests passed")
