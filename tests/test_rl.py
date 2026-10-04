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


def test_tabular_save_load_roundtrip(tmp_path):
    from othello.models import load_model
    env = OthelloEnv(6)
    rng = random.Random(0)
    L = TabularLearner(env, 0, method="mc")
    for _ in range(20):                       # a few short random "episodes"
        s = env.initial_state()
        traj = []
        for _ in range(6):
            a = rng.choice(env.legal_moves(s))
            traj.append((s, a))
            s = env.step(env.step(s, a), rng.choice(env.legal_moves(env.step(s, a))))
        L.end_episode(traj, [0.0] * 5 + [rng.choice([-1.0, 1.0])])
    (tmp_path / "config.json").write_text('{"board_size": 6, "run": "mc-control"}')
    L.save(tmp_path / L.save_name)
    agent = load_model(tmp_path, env)
    L2 = agent.learner
    assert L2.q == L.q and L2.visits == L.visits and L2.states == L.states and L2.method == "mc"
    s = env.initial_state()
    for _ in range(10):                       # greedy choices agree wherever values differ
        _, v1 = L._values(s)
        _, v2 = L2._values(s)
        assert v1 == v2
        s = env.step(s, env.legal_moves(s)[0])
        if env.is_terminal(s):
            break
