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


# ---------------------------------------------------------------- league
class UniformSampler:
    """Every member of a group is equally likely. A PFSP sampler (AlphaStar) can
    replace this later: it gets the same per-opponent W/D/L counts via `counts`."""

    name = "uniform"

    def choose(self, rng, names, counts):
        return rng.choice(names)


class League:
    """Opponents for one training game: the live learner, or a league member.

    With probability `p_league` the opponent is a league member, otherwise the
    live learner (pure self-play). League sampling is two-level, so the league's
    make-up stays constant over training: with probability `fixed_frac` one of
    the fixed agents, otherwise one of the learner's frozen snapshots (all
    league games go to fixed agents until the first snapshot exists). Within a
    group the `sampler` picks the member.

    Per-opponent results are recorded through `record()` and reported (then
    reset) by `stats()` for each evaluation window. Snapshots are reported as
    one aggregate opponent, "snapshots".
    """

    def __init__(self, learner, fixed: dict, p_league: float, snapshot_every=5_000, max_snapshots=20,
                 fixed_frac=0.5, sampler=None, seed=0):
        self.learner, self.fixed, self.p = learner, dict(fixed), p_league
        self.snapshot_every, self.max_snapshots, self.fixed_frac = snapshot_every, max_snapshots, fixed_frac
        self.sampler = sampler or UniformSampler()
        self.rng = random.Random(seed)
        self.snapshots: list[tuple[str, Agent]] = []
        self.self_agent = SelfAgent(learner)
        self.total = {}                      # name -> [W, D, L] over the whole run (for PFSP)
        self._window = {}
        self._last = None

    def maybe_snapshot(self, games: int):
        if games > 0 and games % self.snapshot_every == 0:
            name = f"snap@{games // 1000}k"
            self.snapshots.append((name, self.learner.snapshot(seed=self.rng.randrange(10**9))))
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots.pop(0)

    def sample(self) -> Agent:
        if self.rng.random() >= self.p:
            self._last = ("self", "self")
            return self.self_agent
        if self.snapshots and self.rng.random() >= self.fixed_frac:
            names = [n for n, _ in self.snapshots]
            name = self.sampler.choose(self.rng, names, self.total)
            self._last = ("snapshot", name)
            return dict(self.snapshots)[name]
        name = self.sampler.choose(self.rng, list(self.fixed), self.total)
        self._last = ("fixed", name)
        return self.fixed[name]

    @property
    def last_kind(self):
        return self._last[0] if self._last else None

    def record(self, learner_margin: int):
        """Result of the game just played, as the learner's final disc margin."""
        kind, name = self._last
        idx = 0 if learner_margin > 0 else 1 if learner_margin == 0 else 2
        key = "snapshots" if kind == "snapshot" else name
        for table, k in ((self._window, key), (self.total, name)):
            table.setdefault(k, [0, 0, 0])[idx] += 1

    def stats(self):
        out = {}
        n_all = sum(sum(v) for v in self._window.values())
        for key in ["self", *self.fixed, "snapshots"]:
            w, d, l = self._window.get(key, [0, 0, 0])
            n = w + d + l
            out[f"vs_{key}_n"] = n
            out[f"vs_{key}_score"] = (w + 0.5 * d) / n if n else float("nan")
        n_fixed = sum(sum(self._window.get(k, [0, 0, 0])) for k in self.fixed)
        out["mix_self"] = sum(self._window.get("self", [0, 0, 0])) / n_all if n_all else float("nan")
        out["mix_fixed"] = n_fixed / n_all if n_all else float("nan")
        out["mix_snapshots"] = sum(self._window.get("snapshots", [0, 0, 0])) / n_all if n_all else float("nan")
        out["pool_size"] = len(self.snapshots)
        self._window = {}
        return out


def build_fixed_members(specs: list[dict], env, seed=0) -> dict:
    """League members from specs: {"name", "type": rule-based agent type, "kwargs"} or {"name", "path"}."""
    from othello.agents.registry import make_agent
    from othello.models import load_model
    out = {}
    for i, sp in enumerate(specs):
        if "path" in sp:
            agent = load_model(sp["path"], env)
            if hasattr(agent, "learner"):
                agent.learner.epsilon = 0.0
        else:
            agent = make_agent(sp, env, seed=seed + 101 * (i + 1))
        agent.name = sp["name"]
        out[sp["name"]] = agent
    return out
