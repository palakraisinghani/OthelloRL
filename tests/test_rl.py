import random

from othello.env import OthelloEnv
from othello.rl.tabular import TabularLearner


def _two_step_traj(env):
    s0 = env.initial_state()
    a0 = env.legal_moves(s0)[0]
    s1 = env.step(env.step(s0, a0), env.legal_moves(env.step(s0, a0))[0])
    moves = env.legal_moves(s1)
    return s0, a0, s1, moves


def test_q_and_sarsa_targets_differ_when_next_state_is_known():
    env = OthelloEnv(6)
    s0, a0, s1, moves = _two_step_traj(env)
    taken, other = moves[0], moves[1]
    out = {}
    for method in ("q", "sarsa"):
        L = TabularLearner(env, 0, method=method, alpha=1.0)
        k1, t1 = L._key(s1)
        L.q[L._qkey(k1, t1, other)] = 1.0     # greedy next action is `other`
        L.q[L._qkey(k1, t1, taken)] = -1.0    # but the agent took `taken`
        L.end_episode([(s0, a0), (s1, taken)], [0.0, 0.0])
        k0, t0 = L._key(s0)
        out[method] = L.q[L._qkey(k0, t0, a0)]
    assert out["q"] == 1.0 and out["sarsa"] == -1.0


def test_mc_uses_sample_average():
    env = OthelloEnv(6)
    s0 = env.initial_state()
    a0 = env.legal_moves(s0)[0]
    L = TabularLearner(env, 0, method="mc")
    for r in (1.0, -1.0, 1.0):
        L.end_episode([(s0, a0)], [r])
    k, t = L._key(s0)
    assert abs(L.q[L._qkey(k, t, a0)] - 1 / 3) < 1e-9


def test_symmetric_positions_share_entries():
    env = OthelloEnv(6)
    L = TabularLearner(env, 0, method="q")
    s0 = env.initial_state()
    keys = {L._key(env.step(s0, a))[0] for a in env.legal_moves(s0)}
    assert len(keys) == 1   # the 4 opening moves are symmetric
