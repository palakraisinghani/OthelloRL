"""Tabular control: Q-learning, SARSA and first-visit Monte Carlo.

The table is keyed on the canonical form of the position (from the side to
move's perspective) under the 8 board symmetries, so symmetric positions share
one entry. That is the most favourable setting a lookup table can be given.
Entries are stored in a single flat dict {(canon_key << 7) | canon_action: q};
unseen entries read as 0.
"""
from __future__ import annotations

from othello.rl.common import Learner
from othello.symmetry import get_symmetry


class TabularLearner(Learner):
    name = "tabular"

    def __init__(self, env, seed, method="q", alpha=0.1, gamma=1.0):
        super().__init__(env, seed)
        assert method in ("q", "sarsa", "mc")
        self.method, self.alpha, self.gamma = method, alpha, gamma
        self.name = {"q": "q-learning", "sarsa": "sarsa", "mc": "mc-control"}[method]
        self.sym = get_symmetry(env.n)
        self.shift = 2 * env.n * env.n
        self.q: dict[int, float] = {}
        self.visits: dict[int, int] = {}
        self.states: set[int] = set()
        self._err_n = 0
        self._err_sum = 0.0
        self._err_sq = 0.0

    def _key(self, state):
        (own, opp), t = self.sym.canonical(state.own(), state.opp())
        return (own << (self.env.n * self.env.n)) | opp, t

    def _qkey(self, skey, t, action):
        return (skey << 7) | self.sym.transform_action(action, t)

    def _values(self, state):
        skey, t = self._key(state)
        moves = self.env.legal_moves(state)
        return moves, [self.q.get(self._qkey(skey, t, a), 0.0) for a in moves]

    def act(self, state, explore):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        _, vals = self._values(state)
        best = max(vals)
        return self.rng.choice([m for m, v in zip(moves, vals) if v == best])

    def probe(self, state):
        return self._key(state)[0] in self.states

    def _max_q(self, state):
        if self.env.is_terminal(state):
            return 0.0
        return max(self._values(state)[1])

    def end_episode(self, traj, rewards):
        keys = []
        for s, a in traj:
            skey, t = self._key(s)
            self.states.add(skey)
            keys.append(self._qkey(skey, t, a))
        T = len(traj)
        if self.method == "mc":
            G = 0.0
            returns = [0.0] * T
            for t in reversed(range(T)):
                G = rewards[t] + self.gamma * G
                returns[t] = G
        for t in range(T):
            k = keys[t]
            if self.method == "mc":
                target = returns[t]
            elif t == T - 1:
                target = rewards[t]
            elif self.method == "q":
                target = rewards[t] + self.gamma * self._max_q(traj[t + 1][0])
            else:  # sarsa: the action actually taken next
                target = rewards[t] + self.gamma * self.q.get(keys[t + 1], 0.0)
            old = self.q.get(k, 0.0)
            if self.method == "mc":
                n = self.visits.get(k, 0) + 1
                self.visits[k] = n
                step = 1.0 / n           # sample-average Monte Carlo
            else:
                step = self.alpha
            err = target - old
            self.q[k] = old + step * err
            self._err_n += 1
            self._err_sum += err
            self._err_sq += err * err

    def stats(self):
        n = max(1, self._err_n)
        mean = self._err_sum / n
        out = {"table_states": len(self.states), "table_entries": len(self.q),
               "update_err_var": self._err_sq / n - mean * mean,
               "update_err_rms": (self._err_sq / n) ** 0.5}
        self._err_n, self._err_sum, self._err_sq = 0, 0.0, 0.0
        return out
