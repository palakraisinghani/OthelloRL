"""Shared training loop for learners trained against a fixed opponent.

A learner sees the game only at its own decision points. One episode gives a
trajectory [(s_0, a_0), ..., (s_{T-1}, a_{T-1})] of the learner's states and
actions, the opponent's replies being part of the environment, plus a terminal
reward. Agent states never repeat within an Othello game (discs only
accumulate), so applying the per-step updates at the end of the episode, in
forward order, is identical to updating online.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

from othello.agents.base import Agent
from othello.agents.registry import make_agent
from othello.env import OthelloEnv
from othello.eval.arena import make_openings


class Learner:
    name = "learner"

    def __init__(self, env: OthelloEnv, seed: int):
        self.env = env
        self.rng = random.Random(seed)
        self.epsilon = 0.0

    def act(self, state, explore: bool) -> int:
        raise NotImplementedError

    def end_episode(self, traj, rewards):
        """traj: [(state, action)]; rewards[t] is the reward after step t."""

    def stats(self) -> dict:
        return {}

    def probe(self, state) -> bool | None:
        """For tabular learners: does the table contain `state`? (coverage)"""
        return None


class GreedyAgent(Agent):
    def __init__(self, learner: Learner):
        super().__init__(learner.env)
        self.learner = learner
        self.hits = 0
        self.seen = 0

    def act(self, state):
        p = self.learner.probe(state)
        if p is not None:
            self.seen += 1
            self.hits += p
        return self.learner.act(state, explore=False)


def play(env, opening, black, white):
    s = env.initial_state()
    for m in opening:
        s = env.step(s, m)
    while not env.is_terminal(s):
        agent = black if s.player == 0 else white
        s = env.step(s, agent.act(s))
    return s


def evaluate(learner: Learner, opp: Agent, openings) -> dict:
    env = learner.env
    me = GreedyAgent(learner)
    w = d = margin = 0
    for op in openings:
        for colour in (0, 1):
            black, white = (me, opp) if colour == 0 else (opp, me)
            s = play(env, op, black, white)
            diff = env.score(s) * (1 if colour == 0 else -1)
            w += diff > 0
            d += diff == 0
            margin += diff
    n = 2 * len(openings)
    out = {"win_rate": w / n, "score": (w + 0.5 * d) / n, "draw_rate": d / n, "margin": margin / n}
    if me.seen:
        out["coverage"] = me.hits / me.seen
    return out


def epsilon_at(cfg, ep, total):
    e0, e1, frac = cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_frac"]
    k = min(1.0, ep / max(1, frac * total))
    return e0 + (e1 - e0) * k


def train(learner: Learner, cfg: dict, out_dir: Path, reward_fn=None):
    """Train `learner` against cfg['opponent'] and log learning curves to CSV."""
    env = learner.env
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    rng = random.Random(seed + 7)
    opp = make_agent(cfg["opponent"], env, seed=seed + 11)
    eval_opp = make_agent(cfg["opponent"], env, seed=10_000 + seed)
    eval_openings = make_openings(env, cfg["eval_openings"], cfg["opening_plies"], cfg["eval_seed"])
    total = cfg["train_games"]
    train_openings_pool = make_openings(env, 20_000, cfg["opening_plies"], 1_000 + seed)
    log_path = out_dir / "curve.csv"
    fields = None
    t0 = time.time()
    with log_path.open("w", newline="") as f:
        writer = None
        for ep in range(total + 1):
            if ep % cfg["eval_every"] == 0:
                learner.epsilon = 0.0
                row = {"games": ep, "seconds": time.time() - t0,
                       **evaluate(learner, eval_opp, eval_openings), **learner.stats()}
                if writer is None:
                    fields = list(row)
                    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(learner.name, cfg["seed"], {k: (round(v, 3) if isinstance(v, float) else v)
                                                  for k, v in row.items()}, flush=True)
            if ep == total:
                break
            learner.epsilon = epsilon_at(cfg, ep, total)
            colour = ep % 2
            s = env.initial_state()
            for m in rng.choice(train_openings_pool):
                s = env.step(s, m)
            traj, discs = [], []
            while not env.is_terminal(s):
                if s.player == colour:
                    a = learner.act(s, explore=True)
                    traj.append((s, a))
                    s = env.step(s, a)
                else:
                    s = env.step(s, opp.act(s))
            if not traj:
                continue
            final = env.score(s) * (1 if colour == 0 else -1)
            if reward_fn is None:
                outcome = (final > 0) - (final < 0)
                rewards = [0.0] * (len(traj) - 1) + [float(outcome)]
            else:
                rewards = reward_fn(env, traj, s, colour, final)
            learner.end_episode(traj, rewards)
    return log_path


def final_eval(learner: Learner, cfg: dict, out_dir: Path, opponents: dict):
    env = learner.env
    learner.epsilon = 0.0
    openings = make_openings(env, cfg["final_eval_openings"], cfg["opening_plies"], cfg["eval_seed"] + 1)
    res = {}
    for name, spec in opponents.items():
        n_open = openings if not name.startswith("minimax") else openings[: len(openings) // 2]
        res[name] = evaluate(learner, make_agent(spec, env, seed=20_000 + cfg["seed"]), n_open)
        res[name]["games"] = 2 * len(n_open)
    (out_dir / "final.json").write_text(json.dumps(res, indent=2))
    return res
