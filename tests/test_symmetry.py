import random

import pytest

from othello.env import OthelloEnv, State
from othello.symmetry import get_symmetry


@pytest.mark.parametrize("n", [6, 8])
def test_transforms_commute_with_move_generation(n):
    env, sym = OthelloEnv(n), get_symmetry(n)
    rng = random.Random(1)
    s = env.initial_state()
    for _ in range(15):
        if env.is_terminal(s):
            break
        for t in range(8):
            st = State(sym.transform_bits(s.black, t), sym.transform_bits(s.white, t), s.player)
            assert sym.transform_bits(env.legal_bits(s), t) == env.legal_bits(st)
            for m in env.legal_moves(s):
                if m != env.PASS:
                    a = env.step(s, m)
                    b = env.step(st, sym.transform_action(m, t))
                    assert sym.transform_bits(a.black, t) == b.black
                    assert sym.inverse_action(sym.transform_action(m, t), t) == m
        s = env.step(s, rng.choice(env.legal_moves(s)))


def test_eight_distinct_transforms_and_identity():
    sym = get_symmetry(6)
    assert sym.perms[0] == list(range(36))
    assert len({tuple(p) for p in sym.perms}) == 8


def test_canonical_is_invariant():
    env, sym = OthelloEnv(6), get_symmetry(6)
    rng = random.Random(0)
    s = env.initial_state()
    for _ in range(6):
        s = env.step(s, rng.choice(env.legal_moves(s)))
    key, _ = sym.canonical(s.black, s.white)
    for t in range(8):
        k2, _ = sym.canonical(sym.transform_bits(s.black, t), sym.transform_bits(s.white, t))
        assert k2 == key
