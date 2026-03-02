"""
2x2 Rubik's Cube environment (optimized).

Represents the 2x2 Pocket Cube with 24 stickers (6 faces × 4 stickers).
Uses quarter-turn metric with 6 moves: R, R', U, U', F, F'.
The DBL (Down-Back-Left) corner is fixed so only 3 faces are movable,
giving ~3.67 million reachable states.

State encoding: flat numpy array of 24 integers (0–5), one per sticker color.
One-hot encoding: 144-dim vector (24 positions × 6 colors) for neural network input.

Optimized for batch operations — all core functions are fully vectorized
using numpy broadcasting (no Python loops in hot paths).
"""

import numpy as np


# ── Face layout ──────────────────────────────────────────────────────────────
#
#  Each face has 4 stickers indexed as:
#       0  1
#       2  3
#
#  Faces (in order of the 24-element state array):
#    indices  0– 3 : U (Up)        color 0
#    indices  4– 7 : D (Down)      color 1
#    indices  8–11 : F (Front)     color 2
#    indices 12–15 : B (Back)      color 3
#    indices 16–19 : L (Left)      color 4
#    indices 20–23 : R (Right)     color 5
#

# Solved state
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

# ── Move definitions ─────────────────────────────────────────────────────────
#
# Each move is a permutation on the 24-sticker array.
# We define R, U, F (clockwise 90°) and their inverses (counter-clockwise 90°).
#
# Face indexing reminder (sticker positions within each face):
#   0 1        UL UR        top-left  top-right
#   2 3        DL DR        bot-left  bot-right

# ── R (Right face clockwise, looking at R face) ──
_R_PERM = list(range(24))
_R_PERM[20], _R_PERM[21], _R_PERM[23], _R_PERM[22] = 22, 20, 21, 23
_R_PERM[1], _R_PERM[3] = 9, 11      # U → F
_R_PERM[9], _R_PERM[11] = 5, 7      # F → D
_R_PERM[5], _R_PERM[7] = 14, 12     # D → B
_R_PERM[14], _R_PERM[12] = 1, 3     # B → U

# ── U (Up face clockwise, looking at U face from above) ──
_U_PERM = list(range(24))
_U_PERM[0], _U_PERM[1], _U_PERM[3], _U_PERM[2] = 2, 0, 1, 3
_U_PERM[8], _U_PERM[9] = 20, 21      # F → R
_U_PERM[20], _U_PERM[21] = 12, 13    # R → B
_U_PERM[12], _U_PERM[13] = 16, 17    # B → L
_U_PERM[16], _U_PERM[17] = 8, 9      # L → F

# ── F (Front face clockwise, looking at F face) ──
_F_PERM = list(range(24))
_F_PERM[8], _F_PERM[9], _F_PERM[11], _F_PERM[10] = 10, 8, 9, 11
_F_PERM[2], _F_PERM[3] = 20, 22      # U → R
_F_PERM[20], _F_PERM[22] = 5, 4      # R → D
_F_PERM[4], _F_PERM[5] = 19, 17      # D → L
_F_PERM[17], _F_PERM[19] = 2, 3      # L → U


def _invert_perm(perm):
    """Compute the inverse of a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


# Move table: index → permutation array
# 0: R   1: R'   2: U   3: U'   4: F   5: F'
MOVE_PERMS = [
    np.array(_R_PERM, dtype=np.int8),
    np.array(_invert_perm(_R_PERM), dtype=np.int8),
    np.array(_U_PERM, dtype=np.int8),
    np.array(_invert_perm(_U_PERM), dtype=np.int8),
    np.array(_F_PERM, dtype=np.int8),
    np.array(_invert_perm(_F_PERM), dtype=np.int8),
]

# Pre-stack as (NUM_MOVES, 24) array for vectorized batch operations
MOVE_PERMS_ARRAY = np.stack(MOVE_PERMS)  # shape (6, 24)

MOVE_NAMES = ["R", "R'", "U", "U'", "F", "F'"]
NUM_MOVES = len(MOVE_PERMS)


# ── Core functions ────────────────────────────────────────────────────────────

def apply_move(state: np.ndarray, move_idx: int) -> np.ndarray:
    """Apply a move to a state and return the new state."""
    return state[MOVE_PERMS[move_idx]]


def is_solved(state: np.ndarray) -> bool:
    """Check if the cube is in the solved state."""
    return np.array_equal(state, SOLVED_STATE)


def scramble(n: int, state: np.ndarray = None) -> tuple[np.ndarray, list[int]]:
    """
    Scramble the cube with n random moves.

    Args:
        n: Number of random moves to apply.
        state: Starting state (default: solved).

    Returns:
        (scrambled_state, list_of_move_indices)
    """
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


def get_neighbors(state: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Return all states reachable by one move, with move index."""
    return [(apply_move(state, m), m) for m in range(NUM_MOVES)]


# ── Vectorized batch operations (no Python loops) ────────────────────────────

def batch_scramble(batch_size: int, max_depth: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of scrambled states — fully vectorized.

    All states start from solved and get scrambled in parallel using
    numpy fancy indexing (no Python loop over the batch dimension).

    Args:
        batch_size: Number of scrambled states to generate.
        max_depth: Max scramble depth (each sample uses depth ~ U(1, max_depth)).

    Returns:
        states: (batch_size, 24) int8 array
        depths: (batch_size,) int array of actual depths used
    """
    depths = np.random.randint(1, max_depth + 1, size=batch_size)
    actual_max = int(depths.max())

    # Start all from solved: (batch_size, 24)
    states = np.tile(SOLVED_STATE, (batch_size, 1))

    # Pre-generate all random moves: (actual_max, batch_size)
    all_moves = np.random.randint(0, NUM_MOVES, size=(actual_max, batch_size))

    for step in range(actual_max):
        # Mask: which samples still need a move at this step
        active = step < depths  # (batch_size,) bool

        if not active.any():
            break

        # Get move indices for active samples
        move_indices = all_moves[step]  # (batch_size,)

        # Apply moves to all active samples at once using fancy indexing
        for m in range(NUM_MOVES):
            mask = active & (move_indices == m)
            if mask.any():
                states[mask] = states[mask][:, MOVE_PERMS[m]]

    return states, depths


def batch_get_all_neighbors(states: np.ndarray) -> np.ndarray:
    """
    Get all neighbors for a batch of states — fully vectorized.

    Args:
        states: (N, 24) batch of states

    Returns:
        neighbors: (N, NUM_MOVES, 24) — all neighbor states
    """
    N = states.shape[0]
    # Use pre-stacked permutation array for vectorized indexing
    # states[:, MOVE_PERMS_ARRAY[m]] gives all states after move m
    neighbors = np.empty((N, NUM_MOVES, 24), dtype=np.int8)
    for m in range(NUM_MOVES):
        neighbors[:, m, :] = states[:, MOVE_PERMS[m]]
    return neighbors


def batch_is_solved(states: np.ndarray) -> np.ndarray:
    """
    Check which states in a batch are solved — vectorized.

    Args:
        states: (N, 24) batch of states

    Returns:
        (N,) bool array
    """
    return np.all(states == SOLVED_STATE, axis=1)


def state_to_onehot(state: np.ndarray) -> np.ndarray:
    """
    Convert a 24-element state to a 144-dim one-hot vector.

    Each sticker position gets a 6-dim one-hot for its color.
    """
    onehot = np.zeros(ONE_HOT_DIM, dtype=np.float32)
    for i, color in enumerate(state):
        onehot[i * NUM_COLORS + int(color)] = 1.0
    return onehot


def batch_state_to_onehot(states: np.ndarray) -> np.ndarray:
    """
    Convert a batch of states (N, 24) to one-hot (N, 144).
    Fully vectorized — no Python loops.
    """
    N = states.shape[0]
    onehot = np.zeros((N, ONE_HOT_DIM), dtype=np.float32)
    # Row indices: each row repeated 24 times
    rows = np.repeat(np.arange(N), NUM_STICKERS)
    # Column indices: position * 6 + color
    positions = np.tile(np.arange(NUM_STICKERS), N)
    colors = states.reshape(-1).astype(np.int32)
    cols = positions * NUM_COLORS + colors
    onehot[rows, cols] = 1.0
    return onehot


def state_to_bytes(state: np.ndarray) -> bytes:
    """Convert state to hashable bytes (for use in sets/dicts)."""
    return state.tobytes()


# ── Convenience class ─────────────────────────────────────────────────────────

class Cube2x2:
    """Object-oriented wrapper around the 2x2 cube state functions."""

    def __init__(self, state: np.ndarray = None):
        self.state = state.copy() if state is not None else SOLVED_STATE.copy()

    def apply_move(self, move_idx: int):
        """Apply move in-place."""
        self.state = apply_move(self.state, move_idx)

    def is_solved(self) -> bool:
        return is_solved(self.state)

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def scramble(self, n: int) -> np.ndarray:
        """Scramble in-place, return the scrambled state."""
        self.state, _ = scramble(n, self.state)
        return self.get_state()

    @staticmethod
    def state_to_onehot(state: np.ndarray) -> np.ndarray:
        return state_to_onehot(state)

    def __repr__(self):
        faces = ["U", "D", "F", "B", "L", "R"]
        lines = []
        for i, f in enumerate(faces):
            s = self.state[i * 4 : (i + 1) * 4]
            lines.append(f"{f}: {s}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity checks
    c = Cube2x2()
    assert c.is_solved(), "Fresh cube should be solved"

    # Apply R then R'
    c.apply_move(0)
    assert not c.is_solved(), "After R, should not be solved"
    c.apply_move(1)
    assert c.is_solved(), "After R R', should be solved"

    # Apply U then U'
    c.apply_move(2)
    assert not c.is_solved()
    c.apply_move(3)
    assert c.is_solved()

    # Apply F then F'
    c.apply_move(4)
    assert not c.is_solved()
    c.apply_move(5)
    assert c.is_solved()

    # 4× same move = identity
    for move_idx in range(NUM_MOVES):
        c2 = Cube2x2()
        for _ in range(4):
            c2.apply_move(move_idx)
        assert c2.is_solved(), f"4× move {MOVE_NAMES[move_idx]} should be identity"

    # One-hot encoding shape
    oh = state_to_onehot(c.get_state())
    assert oh.shape == (144,), f"Expected (144,), got {oh.shape}"
    assert oh.sum() == 24, f"Expected 24 ones, got {oh.sum()}"

    # Batch one-hot
    states = np.stack([SOLVED_STATE, SOLVED_STATE])
    boh = batch_state_to_onehot(states)
    assert boh.shape == (2, 144)

    # Batch scramble
    bs, bd = batch_scramble(100, 10)
    assert bs.shape == (100, 24)
    assert bd.shape == (100,)
    # At least some should not be solved
    assert not np.all(batch_is_solved(bs))

    # Batch neighbors
    nbrs = batch_get_all_neighbors(bs[:5])
    assert nbrs.shape == (5, NUM_MOVES, 24)

    print("All cube_env sanity checks passed ✓")
