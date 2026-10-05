import random

from othello.env import OthelloEnv
from othello.rl.population import play_pair


class Recorder:
    """Random learner that records whose perspective each recorded state has."""

    def __init__(self, env, seed):
        self.env, self.rng, self.epsilon = env, random.Random(seed), 0.0

    def act(self, state, explore):
        return self.rng.choice(self.env.legal_moves(state))


def test_play_pair_trajectories_belong_to_the_right_player():
    env = OthelloEnv(6)
    a, b = Recorder(env, 0), Recorder(env, 1)
    for a_black in (True, False):
        ta, tb, margin_a = play_pair(env, a, b, [], a_is_black=a_black)
        assert ta and tb
        assert all(s.player == (0 if a_black else 1) for s, _ in ta)
        assert all(s.player == (1 if a_black else 0) for s, _ in tb)
        assert all(m in env.legal_moves(s) for s, m in ta + tb)
        assert isinstance(margin_a, int)


def test_uniform_pairing_is_balanced_and_never_self():
    rng = random.Random(0)
    n, games = 8, 40_000
    count = [0] * n
    for _ in range(games):
        i, j = rng.sample(range(n), 2)
        assert i != j
        count[i] += 1
        count[j] += 1
    expected = 2 * games / n
    assert all(abs(c - expected) / expected < 0.02 for c in count)


def test_cycle_counting_on_toy_matrices():
    from othello.eval.population_metrics import count_cycles
    names = ["A", "B", "C"]
    games = {(a, b): 1000 for a in names for b in names if a < b}
    rps = {("A", "B"): 0.7, ("B", "C"): 0.7, ("A", "C"): 0.3}       # A>B, B>C, C>A
    cycles, n_beats = count_cycles(names, rps, games)
    assert cycles == [("A", "B", "C")] and n_beats == 3
    transitive = {("A", "B"): 0.7, ("B", "C"): 0.7, ("A", "C"): 0.8}
    assert count_cycles(names, transitive, games)[0] == []
    weak = {("A", "B"): 0.51, ("B", "C"): 0.51, ("A", "C"): 0.49}  # not significant at 1000 games
    assert count_cycles(names, weak, games) == ([], 0)


def test_disagreement_metric():
    from othello.eval.population_metrics import disagreement, mean_pairwise_disagreement
    assert disagreement([1, 2, 3, 4], [1, 2, 0, 0]) == 0.5
    mean, vals = mean_pairwise_disagreement([[1, 1], [1, 2], [2, 2]])
    assert vals == [0.5, 1.0, 0.5] and abs(mean - 2 / 3) < 1e-12
