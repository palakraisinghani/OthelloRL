import random

import pytest

from othello.agents.edax import EDAX_BIN
from othello.env import OthelloEnv

pytestmark = pytest.mark.skipif(not EDAX_BIN.exists(), reason="Edax not built (third_party/edax)")


def test_edax_plays_legal_moves_and_beats_random():
    from othello.agents import RandomAgent
    from othello.agents.edax import EdaxAgent
    from othello.eval.arena import play_game
    env = OthelloEnv(8)
    edax = EdaxAgent(env, level=2)
    rnd = RandomAgent(env, seed=0)
    wins = 0
    for g in range(4):
        black, white = (edax, rnd) if g % 2 == 0 else (rnd, edax)
        diff, moves = play_game(env, black, white, [])     # play_game checks legality via env.step
        wins += (diff > 0) == (g % 2 == 0) and diff != 0
    edax.close()
    assert wins >= 3


def test_edax_board_orientation_matches_engine():
    from othello.agents.edax import EdaxAgent
    env = OthelloEnv(8)
    edax = EdaxAgent(env, level=1)
    rng = random.Random(0)
    s = env.initial_state()
    for _ in range(12):                    # Edax's choice must be legal in our engine at every step
        m = edax.act(s)
        assert m in env.legal_moves(s)
        s = env.step(s, rng.choice(env.legal_moves(s)))
    edax.close()
