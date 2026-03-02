"""
2x2 Rubik's Cube environment.

24 stickers (6 faces x 4 stickers each), quarter-turn metric.
6 moves: R, R', U, U', F, F' (DBL corner fixed).
~3.67M reachable states. God's number = 14.
"""

import numpy as np


# Each face has 4 stickers:
#   0 1
#   2 3
#
# State array layout (24 elements):
#   0-3: U   4-7: D   8-11: F   12-15: B   16-19: L   20-23: R

SOLVED_STATE = np.array(
    [0, 0, 0, 0,   # U
     1, 1, 1, 1,   # D
     2, 2, 2, 2,   # F
     3, 3, 3, 3,   # B
     4, 4, 4, 4,   # L
     5, 5, 5, 5],  # R
    dtype=np.int8,
)

NUM_STICKERS = 24
NUM_COLORS = 6
ONE_HOT_DIM = NUM_STICKERS * NUM_COLORS  # 144

# Move permutations
# Each move is just a permutation of the 24 sticker indices.

# R (right face CW)
_R_PERM = list(range(24))
_R_PERM[20], _R_PERM[21], _R_PERM[23], _R_PERM[22] = 22, 20, 21, 23
_R_PERM[1], _R_PERM[3] = 9, 11      # U -> F
_R_PERM[9], _R_PERM[11] = 5, 7      # F -> D
_R_PERM[5], _R_PERM[7] = 14, 12     # D -> B
_R_PERM[14], _R_PERM[12] = 1, 3     # B -> U

# U (up face CW)
_U_PERM = list(range(24))
_U_PERM[0], _U_PERM[1], _U_PERM[3], _U_PERM[2] = 2, 0, 1, 3
_U_PERM[8], _U_PERM[9] = 20, 21      # F -> R
_U_PERM[20], _U_PERM[21] = 12, 13    # R -> B
_U_PERM[12], _U_PERM[13] = 16, 17    # B -> L
_U_PERM[16], _U_PERM[17] = 8, 9      # L -> F

# F (front face CW)
_F_PERM = list(range(24))
_F_PERM[8], _F_PERM[9], _F_PERM[11], _F_PERM[10] = 10, 8, 9, 11
_F_PERM[2], _F_PERM[3] = 20, 22      # U -> R
_F_PERM[20], _F_PERM[22] = 5, 4      # R -> D
_F_PERM[4], _F_PERM[5] = 19, 17      # D -> L
_F_PERM[17], _F_PERM[19] = 2, 3      # L -> U


def _invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


# 0: R, 1: R', 2: U, 3: U', 4: F, 5: F'
MOVE_PERMS = [
    np.array(_R_PERM, dtype=np.int8),
    np.array(_invert_perm(_R_PERM), dtype=np.int8),
    np.array(_U_PERM, dtype=np.int8),
    np.array(_invert_perm(_U_PERM), dtype=np.int8),
    np.array(_F_PERM, dtype=np.int8),
    np.array(_invert_perm(_F_PERM), dtype=np.int8),
]

MOVE_PERMS_ARRAY = np.stack(MOVE_PERMS)  # (6, 24) for batch ops

MOVE_NAMES = ["R", "R'", "U", "U'", "F", "F'"]
NUM_MOVES = len(MOVE_PERMS)


def apply_move(state, move_idx):
    return state[MOVE_PERMS[move_idx]]


def is_solved(state):
    return np.array_equal(state, SOLVED_STATE)


def scramble(n, state=None):
    """Scramble with n random moves. Returns (state, move_list)."""
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


# Batch operations (vectorized with numpy)

def batch_scramble(batch_size, max_depth):
    """Generate batch_size scrambled states. Vectorized over the batch dim."""
    depths = np.random.randint(1, max_depth + 1, size=batch_size)
    actual_max = int(depths.max())
    states = np.tile(SOLVED_STATE, (batch_size, 1))
    all_moves = np.random.randint(0, NUM_MOVES, size=(actual_max, batch_size))

    for step in range(actual_max):
        active = step < depths
        if not active.any():
            break
        move_indices = all_moves[step]
        for m in range(NUM_MOVES):
            mask = active & (move_indices == m)
            if mask.any():
                states[mask] = states[mask][:, MOVE_PERMS[m]]

    return states, depths


def batch_get_all_neighbors(states):
    """(N, 24) -> (N, NUM_MOVES, 24) with all neighbor states."""
    N = states.shape[0]
    neighbors = np.empty((N, NUM_MOVES, 24), dtype=np.int8)
    for m in range(NUM_MOVES):
        neighbors[:, m, :] = states[:, MOVE_PERMS[m]]
    return neighbors


def batch_is_solved(states):
    """(N, 24) -> (N,) bool array."""
    return np.all(states == SOLVED_STATE, axis=1)


def state_to_onehot(state):
    """Single state -> 144-dim one-hot vector."""
    onehot = np.zeros(ONE_HOT_DIM, dtype=np.float32)
    for i, color in enumerate(state):
        onehot[i * NUM_COLORS + int(color)] = 1.0
    return onehot


def batch_state_to_onehot(states):
    """(N, 24) -> (N, 144) one-hot. Vectorized."""
    N = states.shape[0]
    onehot = np.zeros((N, ONE_HOT_DIM), dtype=np.float32)
    rows = np.repeat(np.arange(N), NUM_STICKERS)
    positions = np.tile(np.arange(NUM_STICKERS), N)
    colors = states.reshape(-1).astype(np.int32)
    cols = positions * NUM_COLORS + colors
    onehot[rows, cols] = 1.0
    return onehot


def state_to_bytes(state):
    return state.tobytes()


class Cube2x2:
    """OOP wrapper if you want it."""

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
    # sanity checks
    c = Cube2x2()
    assert c.is_solved()

    c.apply_move(0)  # R
    assert not c.is_solved()
    c.apply_move(1)  # R'
    assert c.is_solved()

    c.apply_move(2)  # U
    assert not c.is_solved()
    c.apply_move(3)  # U'
    assert c.is_solved()

    c.apply_move(4)  # F
    assert not c.is_solved()
    c.apply_move(5)  # F'
    assert c.is_solved()

    # 4x any move = identity
    for move_idx in range(NUM_MOVES):
        c2 = Cube2x2()
        for _ in range(4):
            c2.apply_move(move_idx)
        assert c2.is_solved(), f"4x {MOVE_NAMES[move_idx]} should be identity"

    # one-hot
    oh = state_to_onehot(c.get_state())
    assert oh.shape == (144,) and oh.sum() == 24

    # batch ops
    states = np.stack([SOLVED_STATE, SOLVED_STATE])
    boh = batch_state_to_onehot(states)
    assert boh.shape == (2, 144)

    bs, bd = batch_scramble(100, 10)
    assert bs.shape == (100, 24)
    assert not np.all(batch_is_solved(bs))

    nbrs = batch_get_all_neighbors(bs[:5])
    assert nbrs.shape == (5, NUM_MOVES, 24)

    print("All cube_env sanity checks passed ✓")
