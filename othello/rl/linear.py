"""Q-learning with linear function approximation over hand-crafted features.

Q(s, a) = w . phi(after(s, a)), where after(s, a) is the position right after
the learner's move (before the opponent replies), seen from the learner's side.
Valuing afterstates is the standard choice for board games: moves that lead to
the same position share one value.
"""
from __future__ import annotations

import numpy as np

from othello.agents.heuristic import Evaluator, HeuristicWeights
from othello.env import State
from othello.rl.common import Learner

ALL_FEATURES = ("bias", "disc", "mobility", "corner", "xsquare", "edge_stable")


class LinearQLearner(Learner):
    name = "linear-q"

    def __init__(self, env, seed, features=ALL_FEATURES, alpha=0.01, gamma=1.0, name=None):
        super().__init__(env, seed)
        self.features = tuple(features)
        self.alpha, self.gamma = alpha, gamma
        self.w = np.zeros(len(self.features))
        self.ev = Evaluator(env, HeuristicWeights())
        if name:
            self.name = name
        n = env.n
        corners = [0, n - 1, n * (n - 1), n * n - 1]
        steps = {0: (1, n), n - 1: (-1, n), n * (n - 1): (1, -n), n * n - 1: (-1, -n)}
        self._edges = [(c, steps[c]) for c in corners]
        nn = n * n
        self._scale = {"bias": 1.0, "disc": 1.0 / nn, "mobility": 1.0 / 10, "corner": 1.0 / 4,
                       "xsquare": 1.0 / 4, "edge_stable": 1.0 / (4 * (n - 1))}
        self._err = []

    def _stable_edges(self, mine: int) -> int:
        """Discs of `mine` on an edge, connected to an own corner by an unbroken own run."""
        n = self.env.n
        stable = 0
        for c, (d1, d2) in self._edges:
            if not mine >> c & 1:
                continue
            for d in (d1, d2):
                sq = c
                for _ in range(n):
                    if not mine >> sq & 1:
                        break
                    stable |= 1 << sq
                    sq += d
        return stable.bit_count()

    def phi(self, after: State, me: int) -> np.ndarray:
        f = self.ev.features(after, me)
        mine = after.black if me == 0 else after.white
        theirs = after.white if me == 0 else after.black
        f["edge_stable"] = self._stable_edges(mine) - self._stable_edges(theirs)
        f["bias"] = 1.0
        return np.array([f[k] * self._scale[k] for k in self.features])

    def _phis(self, state):
        moves = self.env.legal_moves(state)
        return moves, [self.phi(self.env.step(state, a), state.player) for a in moves]

    def act(self, state, explore):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        _, ph = self._phis(state)
        vals = [float(self.w @ p) for p in ph]
        best = max(vals)
        return self.rng.choice([m for m, v in zip(moves, vals) if v == best])

    def end_episode(self, traj, rewards):
        T = len(traj)
        for t, (s, a) in enumerate(traj):
            p = self.phi(self.env.step(s, a), s.player)
            if t == T - 1:
                target = rewards[t]
            else:
                _, ph = self._phis(traj[t + 1][0])
                target = rewards[t] + self.gamma * max(float(self.w @ q) for q in ph)
            err = target - float(self.w @ p)
            self.w += self.alpha * err * p
            self._err.append(err)

    def stats(self):
        e = np.array(self._err) if self._err else np.zeros(1)
        self._err = []
        return {"update_err_rms": float(np.sqrt((e ** 2).mean())),
                **{f"w_{k}": float(v) for k, v in zip(self.features, self.w)}}


def disc_margin_reward(env, traj, final_state, colour, final_margin):
    """'Maximise discs' objective: the terminal reward is the final disc margin."""
    return [0.0] * (len(traj) - 1) + [final_margin / (env.n * env.n)]
