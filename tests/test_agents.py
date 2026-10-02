import pytest

from othello.agents import HeuristicAgent, MinimaxAgent, RandomAgent
from othello.agents.heuristic import Evaluator, HeuristicWeights
from othello.eval.arena import make_openings, play_match
from othello.eval.elo import fit_elo
from othello.env import OthelloEnv, State


@pytest.mark.parametrize("cls", [RandomAgent, HeuristicAgent, MinimaxAgent])
@pytest.mark.parametrize("n", [6, 8])
def test_agents_only_play_legal_moves(cls, n):
    env = OthelloEnv(n)
    agent = cls(env, seed=0) if cls is not MinimaxAgent else cls(env, depth=2, seed=0)
    s = env.initial_state()
    while not env.is_terminal(s):
        m = agent.act(s)
        assert m in env.legal_moves(s)
        s = env.step(s, m)


def test_heuristic_takes_corner():
    env = OthelloEnv(4)
    bit = lambda r, c: 1 << (r * 4 + c)
    # Black can play a1 (corner, flips b2) or other squares; corner weight dominates.
    s = State(black=bit(2, 2), white=bit(1, 1) | bit(1, 2), player=0)
    agent = HeuristicAgent(env, HeuristicWeights(disc=0, mobility=0, corner=10), seed=0)
    assert agent.act(s) == 0


def test_minimax_finds_forced_win():
    """Depth-2 search must see a move that wipes out the opponent."""
    env = OthelloEnv(4)
    bit = lambda r, c: 1 << (r * 4 + c)
    s = State(black=bit(0, 0), white=bit(0, 1), player=0)  # b1 captured by c1 -> white has no discs
    agent = MinimaxAgent(env, depth=2, weights=HeuristicWeights(disc=0, mobility=0, corner=0), seed=0)
    assert agent.act(s) == 2


def test_evaluator_is_zero_sum_on_start():
    env = OthelloEnv(6)
    ev = Evaluator(env, HeuristicWeights())
    s = env.initial_state()
    assert ev.evaluate(s, 0) == -ev.evaluate(s, 1) == 0


def test_openings_are_deterministic():
    env = OthelloEnv(6)
    assert make_openings(env, 5, 4, seed=3) == make_openings(env, 5, 4, seed=3)


def test_match_plays_both_colours():
    r = play_match(6, {"type": "random"}, {"type": "random"}, n_openings=5, workers=1, keep_games=True)
    assert r.n == 10
    assert sum(g["a_is_black"] for g in r.games) == 5


def test_elo_orders_players():
    elo = fit_elo({("a", "b"): (90, 100), ("b", "c"): (90, 100), ("a", "c"): (99, 100)}, anchor="c")
    assert elo["a"] > elo["b"] > elo["c"] == 0
