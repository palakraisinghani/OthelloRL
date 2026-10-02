"""Opponent pool for self-play.

Every `snapshot_every` games the learner's current policy is frozen and added
to the pool (oldest dropped beyond `max_size`). Each training game is played
against the learner itself with probability `p_latest`, otherwise against a
uniformly chosen past snapshot. Playing past selves keeps the learner from
forgetting how to beat earlier strategies (strategy cycling), which pure
latest-vs-latest self-play is prone to.
"""
from __future__ import annotations

import random

from othello.agents.base import Agent


class SelfAgent(Agent):
    """The live learner as an opponent (exploring, like the learner itself)."""

    def __init__(self, learner):
        super().__init__(learner.env)
        self.learner = learner

    def act(self, state):
        return self.learner.act(state, explore=True)


class OpponentPool:
    def __init__(self, learner, snapshot_every=5_000, max_size=20, p_latest=0.5, seed=0):
        self.learner = learner
        self.snapshot_every, self.max_size, self.p_latest = snapshot_every, max_size, p_latest
        self.rng = random.Random(seed)
        self.snapshots: list[Agent] = []
        self.self_agent = SelfAgent(learner)
        self.n_latest = self.n_past = 0

    def maybe_snapshot(self, games: int):
        if games % self.snapshot_every == 0:
            self.snapshots.append(self.learner.snapshot(seed=self.rng.randrange(10**9)))
            if len(self.snapshots) > self.max_size:
                self.snapshots.pop(0)

    def sample(self) -> Agent:
        if not self.snapshots or self.rng.random() < self.p_latest:
            self.n_latest += 1
            return self.self_agent
        self.n_past += 1
        return self.rng.choice(self.snapshots)

    def stats(self):
        out = {"pool_size": len(self.snapshots),
               "frac_vs_past": self.n_past / max(1, self.n_past + self.n_latest)}
        self.n_latest = self.n_past = 0
        return out
