"""Size-parameterised Othello environment on Python-int bitboards.

Square (r, c) maps to bit r * n + c. Actions are square indices 0..n*n-1,
and PASS = n*n is the forced-pass action, legal only when no placement is.
States are immutable, so cloning is free and MCTS can keep references.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLACK, WHITE = 0, 1
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class State:
    black: int
    white: int
    player: int        # side to move: BLACK or WHITE
    passes: int = 0    # consecutive passes so far (2 => game over)

    def own(self) -> int:
        return self.black if self.player == BLACK else self.white

    def opp(self) -> int:
        return self.white if self.player == BLACK else self.black


class OthelloEnv:
    def __init__(self, board_size: int = 6):
        if board_size < 4 or board_size % 2:
            raise ValueError("board_size must be an even integer >= 4")
        n = self.n = board_size
        self.num_squares = n * n
        self.PASS = n * n
        self.num_actions = n * n + 1
        self.full = (1 << (n * n)) - 1
        col0 = sum(1 << (r * n) for r in range(n))
        coln = col0 << (n - 1)
        # (shift amount, mask of squares a shifted bit may legally land on)
        self._shifts = []
        for dr, dc in DIRECTIONS:
            mask = self.full
            if dc == 1:
                mask &= ~col0   # moving right must not wrap into column 0
            elif dc == -1:
                mask &= ~coln   # moving left must not wrap into column n-1
            self._shifts.append((dr * n + dc, mask))

    # ------------------------------------------------------------------ state
    def initial_state(self) -> State:
        n, h = self.n, self.n // 2
        bit = lambda r, c: 1 << (r * n + c)
        white = bit(h - 1, h - 1) | bit(h, h)
        black = bit(h - 1, h) | bit(h, h - 1)
        return State(black=black, white=white, player=BLACK)

    def _shift(self, x: int, s: int, mask: int) -> int:
        return ((x << s) if s > 0 else (x >> -s)) & mask

    def legal_bits(self, state: State) -> int:
        own, opp = state.own(), state.opp()
        empty = self.full & ~(own | opp)
        moves = 0
        for s, mask in self._shifts:
            x = self._shift(own, s, mask) & opp
            for _ in range(self.n - 3):
                x |= self._shift(x, s, mask) & opp
            moves |= self._shift(x, s, mask) & empty
        return moves

    def legal_moves(self, state: State) -> list[int]:
        """Legal actions; [PASS] if the side to move has no placement."""
        if self.is_terminal(state):
            return []
        bits = self.legal_bits(state)
        if not bits:
            return [self.PASS]
        moves = []
        while bits:
            low = bits & -bits
            moves.append(low.bit_length() - 1)
            bits ^= low
        return moves

    def legal_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[self.legal_moves(state)] = True
        return mask

    def _flips(self, own: int, opp: int, sq: int) -> int:
        flips = 0
        m = 1 << sq
        for s, mask in self._shifts:
            line = 0
            x = self._shift(m, s, mask)
            while x & opp:
                line |= x
                x = self._shift(x, s, mask)
            if x & own:
                flips |= line
        return flips

    def step(self, state: State, action: int) -> State:
        """Apply a legal action and return the new state (no legality check on hot path)."""
        nxt = 1 - state.player
        if action == self.PASS:
            return State(state.black, state.white, nxt, state.passes + 1)
        own, opp = state.own(), state.opp()
        f = self._flips(own, opp, action)
        if not f:
            raise ValueError(f"illegal move {action}")
        own |= f | (1 << action)
        opp &= ~f
        if state.player == BLACK:
            return State(own, opp, nxt, 0)
        return State(opp, own, nxt, 0)

    # --------------------------------------------------------------- terminal
    def is_terminal(self, state: State) -> bool:
        return state.passes >= 2 or (state.black | state.white) == self.full

    def counts(self, state: State) -> tuple[int, int]:
        return state.black.bit_count(), state.white.bit_count()

    def score(self, state: State) -> int:
        """Disc difference, black minus white."""
        b, w = self.counts(state)
        return b - w

    def winner(self, state: State):
        """BLACK, WHITE, or None for a draw (call on terminal states)."""
        d = self.score(state)
        return BLACK if d > 0 else WHITE if d < 0 else None

    # --------------------------------------------------------------- encoding
    PLANES = ("mine", "theirs", "empty", "turn", "legal")

    def to_array(self, bits: int) -> np.ndarray:
        n = self.n
        arr = np.array([(bits >> i) & 1 for i in range(n * n)], dtype=np.float32)
        return arr.reshape(n, n)

    def encode(self, state: State, planes=("mine", "theirs")) -> np.ndarray:
        """(C, n, n) float32 tensor of the requested feature planes."""
        out = []
        for p in planes:
            if p == "mine":
                out.append(self.to_array(state.own()))
            elif p == "theirs":
                out.append(self.to_array(state.opp()))
            elif p == "empty":
                out.append(self.to_array(self.full & ~(state.black | state.white)))
            elif p == "turn":
                out.append(np.full((self.n, self.n), float(state.player), np.float32))
            elif p == "legal":
                out.append(self.to_array(self.legal_bits(state)))
            else:
                raise ValueError(f"unknown plane {p!r}")
        return np.stack(out)

    def render(self, state: State) -> str:
        n = self.n
        rows = ["  " + " ".join("abcdefghijklmnop"[:n])]
        for r in range(n):
            cells = []
            for c in range(n):
                b = 1 << (r * n + c)
                cells.append("X" if state.black & b else "O" if state.white & b else ".")
            rows.append(f"{r + 1} " + " ".join(cells))
        rows.append(f"to move: {'X (black)' if state.player == BLACK else 'O (white)'}")
        return "\n".join(rows)

    def action_name(self, action: int) -> str:
        if action == self.PASS:
            return "pass"
        r, c = divmod(action, self.n)
        return "abcdefghijklmnop"[c] + str(r + 1)
