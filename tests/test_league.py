import random

import pytest

from othello.agents import RandomAgent
from othello.env import OthelloEnv
from othello.rl.pool import League


class FakeLearner:
    def __init__(self, env):
        self.env = env

    def act(self, state, explore):
        return random.choice(self.env.legal_moves(state))

    def snapshot(self, seed=0):
        return RandomAgent(self.env, seed=seed)


def make_league(p, snapshots=0, fixed_n=8, seed=0):
    env = OthelloEnv(6)
    fixed = {f"f{i}": RandomAgent(env, seed=i) for i in range(fixed_n)}
    lg = League(FakeLearner(env), fixed, p_league=p, snapshot_every=5_000, seed=seed)
    for g in range(1, snapshots + 1):
        lg.maybe_snapshot(5_000 * g)
    return lg


def shares(lg, n=10_000):
    c = {"self": 0, "fixed": 0, "snapshot": 0}
    for _ in range(n):
        lg.sample()
        c[lg.last_kind] += 1
    return {k: v / n for k, v in c.items()}


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_sampling_frequencies_two_level(p):
    sh = shares(make_league(p, snapshots=5))
    assert abs(sh["self"] - (1 - p)) < 0.02
    assert abs(sh["fixed"] - p / 2) < 0.02
    assert abs(sh["snapshot"] - p / 2) < 0.02


def test_league_games_go_to_fixed_until_first_snapshot():
    sh = shares(make_league(1.0, snapshots=0), n=2_000)
    assert sh["fixed"] == 1.0


def test_fixed_members_sampled_uniformly():
    lg = make_league(1.0, snapshots=0)
    counts = {}
    for _ in range(8_000):
        lg.sample()
        counts[lg._last[1]] = counts.get(lg._last[1], 0) + 1
    assert set(counts) == {f"f{i}" for i in range(8)}
    assert all(abs(c / 8_000 - 1 / 8) < 0.02 for c in counts.values())


def test_snapshot_cap_and_schedule():
    lg = make_league(0.5, snapshots=25)
    assert len(lg.snapshots) == 20
    assert lg.snapshots[0][0] == "snap@30k" and lg.snapshots[-1][0] == "snap@125k"
    lg2 = make_league(0.5)
    lg2.maybe_snapshot(0)
    lg2.maybe_snapshot(4_999)
    assert lg2.snapshots == []


def test_stats_columns_are_stable_and_reset():
    lg = make_league(0.5, snapshots=1)
    empty = lg.stats()
    for _ in range(300):
        lg.sample()
        lg.record(random.choice([-4, 0, 6]))
    full = lg.stats()
    assert list(empty) == list(full)                      # same CSV columns every window
    assert abs(full["mix_self"] + full["mix_fixed"] + full["mix_snapshots"] - 1) < 1e-9
    assert sum(full[f"vs_f{i}_n"] for i in range(8)) + full["vs_self_n"] + full["vs_snapshots_n"] == 300
    assert lg.stats()["vs_self_n"] == 0                   # window was reset
    assert sum(sum(v) for v in lg.total.values()) == 300  # run totals kept (for PFSP)
