"""Co-evolution: N independent learners that train only by playing each other.

Every game pairs two *different* members drawn uniformly at random; both
members act with their own exploration schedule and both learn from their own
moves (one game trains two learners). Members never play a copy of themselves.
Each member keeps its own game counter, which drives its epsilon schedule,
its evaluation/checkpoint timing and its logs, so a member is directly
comparable with a single self-play DQN that played the same number of games.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

from othello.agents.registry import make_agent
from othello.eval.arena import make_openings
from othello.rl.common import epsilon_at, evaluate, final_eval


def play_pair(env, a, b, opening, a_is_black):
    """One training game between learners a and b. Returns (traj_a, traj_b, margin_a)."""
    s = env.initial_state()
    for m in opening:
        s = env.step(s, m)
    colour = {0: a if a_is_black else b, 1: b if a_is_black else a}
    trajs = {id(a): [], id(b): []}
    while not env.is_terminal(s):
        mover = colour[s.player]
        act = mover.act(s, explore=True)
        trajs[id(mover)].append((s, act))
        s = env.step(s, act)
    margin_black = env.score(s)
    margin_a = margin_black if a_is_black else -margin_black
    return trajs[id(a)], trajs[id(b)], margin_a


def _rewards(traj, margin):
    outcome = (margin > 0) - (margin < 0)
    return [0.0] * (len(traj) - 1) + [float(outcome)]


def train_population(learners: list, cfg: dict, out_dir: Path):
    """Train `learners` by uniform pairwise co-evolution for cfg['games_per_member'] games each (in expectation)."""
    n = len(learners)
    assert n >= 2, "a population needs at least two members"
    env = learners[0].env
    seed = cfg["seed"]
    rng = random.Random(seed + 7)
    per_member = cfg["games_per_member"]
    total = n * per_member // 2
    openings = make_openings(env, 20_000, cfg["opening_plies"], 1_000 + seed)
    eval_openings = make_openings(env, cfg["eval_openings"], cfg["opening_plies"], cfg["eval_seed"])
    eval_opp = make_agent(cfg["opponent"], env, seed=10_000 + seed)
    dirs = [out_dir / f"m{i}" for i in range(n)]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    count = [0] * n
    # pairwise results for the current logging window: wdl[i][j] = [W, D, L] of i against j
    window = [[[0, 0, 0] for _ in range(n)] for _ in range(n)]
    files = [open(d / "curve.csv", "w", newline="") for d in dirs]
    writers = [None] * n
    t0 = time.time()

    def log(i):
        learners[i].epsilon = 0.0
        row = {"games": count[i], "total_games": sum(count) // 2, "seconds": time.time() - t0,
               **evaluate(learners[i], eval_opp, eval_openings), **learners[i].stats()}
        for j in range(n):
            if j != i:
                w, d, l = window[i][j]
                tot = w + d + l
                row[f"vs_m{j}_n"] = tot
                row[f"vs_m{j}_score"] = (w + 0.5 * d) / tot if tot else float("nan")
                window[i][j] = [0, 0, 0]
        if writers[i] is None:
            writers[i] = csv.DictWriter(files[i], fieldnames=list(row))
            writers[i].writeheader()
        writers[i].writerow(row)
        files[i].flush()
        print(f"m{i}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                        if k in ("games", "score", "seconds")}, flush=True)

    for i in range(n):
        log(i)
    try:
        for g in range(total):
            i, j = rng.sample(range(n), 2)
            for k in (i, j):
                learners[k].epsilon = epsilon_at(cfg, count[k], per_member)
            ti, tj, mi = play_pair(env, learners[i], learners[j], rng.choice(openings), a_is_black=g % 2 == 0)
            if ti:
                learners[i].end_episode(ti, _rewards(ti, mi))
            if tj:
                learners[j].end_episode(tj, _rewards(tj, -mi))
            idx = 0 if mi > 0 else 1 if mi == 0 else 2
            window[i][j][idx] += 1
            window[j][i][2 - idx] += 1
            for k in (i, j):
                count[k] += 1
                if count[k] % cfg["eval_every"] == 0:
                    log(k)
                ck = cfg.get("checkpoint_every")
                if ck and count[k] % ck == 0 and hasattr(learners[k], "save"):
                    (dirs[k] / "ckpt").mkdir(exist_ok=True)
                    learners[k].save(dirs[k] / "ckpt" / f"g{count[k]:06d}.pt")
    finally:
        for f in files:
            f.close()
    for i in range(n):
        res = final_eval(learners[i], {**cfg, "seed": seed * 100 + i}, dirs[i], cfg["final_opponents"])
        res["games"] = count[i]
        res["train_seconds"] = time.time() - t0
        (dirs[i] / "final.json").write_text(json.dumps(res, indent=2))
        if hasattr(learners[i], "save"):
            learners[i].save(dirs[i] / "model.pt")
    return count
