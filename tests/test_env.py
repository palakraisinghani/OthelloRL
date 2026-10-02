import random

import numpy as np
import pytest

from othello.env import BLACK, WHITE, OthelloEnv, State

SIZES = [4, 6, 8, 10]


def perft(env, s, depth):
    if depth == 0 or env.is_terminal(s):
        return 1
    return sum(perft(env, env.step(s, m), depth - 1) for m in env.legal_moves(s))


def naive_moves_and_flips(n, grid, player):
    """Reference implementation on a 2-D grid (0 empty, 1 black, 2 white)."""
    me, them = (1, 2) if player == BLACK else (2, 1)
    out = {}
    for r in range(n):
        for c in range(n):
            if grid[r][c]:
                continue
            flips = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    line, rr, cc = [], r + dr, c + dc
                    while 0 <= rr < n and 0 <= cc < n and grid[rr][cc] == them:
                        line.append((rr, cc))
                        rr, cc = rr + dr, cc + dc
                    if line and 0 <= rr < n and 0 <= cc < n and grid[rr][cc] == me:
                        flips += line
            if flips:
                out[r * n + c] = set(flips)
    return out


def to_grid(env, s):
    n = env.n
    return [[1 if s.black >> (r * n + c) & 1 else 2 if s.white >> (r * n + c) & 1 else 0
             for c in range(n)] for r in range(n)]


@pytest.mark.parametrize("n", SIZES)
def test_initial_position(n):
    env = OthelloEnv(n)
    s = env.initial_state()
    assert env.counts(s) == (2, 2)
    assert s.player == BLACK
    assert len(env.legal_moves(s)) == 4


def test_perft_8x8_known_values():
    env = OthelloEnv(8)
    s = env.initial_state()
    assert [perft(env, s, d) for d in range(1, 7)] == [4, 12, 56, 244, 1396, 8200]


@pytest.mark.parametrize("n", SIZES)
def test_matches_naive_reference_on_random_games(n):
    env = OthelloEnv(n)
    rng = random.Random(n)
    for _ in range(30):
        s = env.initial_state()
        while not env.is_terminal(s):
            grid = to_grid(env, s)
            ref = naive_moves_and_flips(n, grid, s.player)
            moves = env.legal_moves(s)
            assert set(moves) == (set(ref) or {env.PASS})
            m = rng.choice(moves)
            nxt = env.step(s, m)
            if m != env.PASS:
                me = 1 if s.player == BLACK else 2
                ng = to_grid(env, nxt)
                for (r, c) in ref[m]:
                    assert ng[r][c] == me
                b0, w0 = env.counts(s)
                b1, w1 = env.counts(nxt)
                assert b1 + w1 == b0 + w0 + 1
            s = nxt


def test_forced_pass_and_double_pass_end():
    env = OthelloEnv(4)
    n = 4
    bit = lambda r, c: 1 << (r * n + c)
    # Black's only disc pair cannot capture anything; white can.
    #   . . . .
    #   . X O .
    #   . . . .
    #   . . . .
    s = State(black=bit(1, 1), white=bit(1, 2), player=WHITE)
    assert env.legal_moves(s) == [bit(1, 0).bit_length() - 1]
    s = env.step(s, env.legal_moves(s)[0])         # white captures the only black disc
    assert env.counts(s) == (0, 3)
    assert env.legal_moves(s) == [env.PASS]          # black must pass
    s = env.step(s, env.PASS)
    assert s.passes == 1 and not env.is_terminal(s)
    assert env.legal_moves(s) == [env.PASS]          # white has nothing to flip either
    s = env.step(s, env.PASS)
    assert env.is_terminal(s) and env.winner(s) == WHITE
    assert env.legal_moves(s) == []


def test_pass_counter_resets_after_placement():
    env = OthelloEnv(4)
    bit = lambda r, c: 1 << (r * 4 + c)
    s = State(black=bit(0, 2), white=bit(0, 1), player=BLACK, passes=1)
    assert env.legal_moves(s) == [0]
    assert env.step(s, 0).passes == 0


def test_full_board_terminal_and_draw():
    env = OthelloEnv(4)
    half = sum(1 << i for i in range(8))
    s = State(black=half, white=env.full & ~half, player=BLACK)
    assert env.is_terminal(s)
    assert env.winner(s) is None


def test_illegal_move_raises():
    env = OthelloEnv(6)
    with pytest.raises(ValueError):
        env.step(env.initial_state(), 0)


@pytest.mark.parametrize("n", [6, 8, 10])
def test_dihedral_symmetry_of_move_generation(n):
    """Transposing the board must transpose the legal-move set."""
    env = OthelloEnv(n)
    rng = random.Random(0)
    T = lambda bits: sum(1 << (c * n + r) for r in range(n) for c in range(n) if bits >> (r * n + c) & 1)
    s = env.initial_state()
    for _ in range(12):
        if env.is_terminal(s):
            break
        st = State(T(s.black), T(s.white), s.player)
        assert T(env.legal_bits(s)) == env.legal_bits(st)
        s = env.step(s, rng.choice(env.legal_moves(s)))


@pytest.mark.parametrize("n", [6, 8, 10])
def test_encode_shapes(n):
    env = OthelloEnv(n)
    s = env.initial_state()
    for planes in [("mine", "theirs"), ("mine", "theirs", "empty"),
                   ("mine", "theirs", "empty", "turn"), env.PLANES]:
        x = env.encode(s, planes)
        assert x.shape == (len(planes), n, n) and x.dtype == np.float32
    x = env.encode(s, env.PLANES)
    assert x[0].sum() == 2 and x[1].sum() == 2 and x[2].sum() == n * n - 4 and x[4].sum() == 4
    assert env.legal_mask(s).sum() == 4
