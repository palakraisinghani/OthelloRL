"""Phase 4 evaluation (Steps 4.5-4.6). All stages are resumable.

  .venv/bin/python scripts/eval_coevo.py members   # each member vs self-play DQN, Phase 2 PPO, DQN+
  .venv/bin/python scripts/eval_coevo.py within    # all member pairs inside each population
  .venv/bin/python scripts/eval_coevo.py ckpt      # member checkpoints vs minimax-d3 and PPO seed 0
  .venv/bin/python scripts/eval_coevo.py all

Scores against the rule-based bots come from each member's final.json (written at
the end of training: 1,000 games vs random/heuristic, 500 vs each minimax depth).
"""
import argparse
import json
import shutil
import sys
import tempfile
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.eval.arena import play_match  # noqa: E402
from othello.utils.config import load_json  # noqa: E402

OUT = ROOT / "results/coevo"
SEEDS = [0, 1, 2]
W = load_json(ROOT / "results/baselines/heuristic_tuning.json")["best"]
REFS = {**{f"selfplay-s{k}": f"results/league/dqn/p0/seed{k}" for k in SEEDS},
        **{f"ppo-s{k}": f"results/selfplay/ppo/seed{k}" for k in SEEDS},
        **{f"dqn+-s{k}": f"results/selfplay/dqn-plus/seed{k}" for k in SEEDS}}


def members():
    for run_dir in sorted(OUT.glob("F*/seed*")):
        for m in sorted(run_dir.glob("m*")):
            if (m / "final.json").exists():
                yield f"{run_dir.parent.name}/{run_dir.name}/{m.name}", m


def _spec(x):
    return x if isinstance(x, dict) else {"path": str(x)}


def _job(args):
    """Worker: play one whole match single-process (the pool parallelises across matches)."""
    key, sub, a, b, n_open, seed = args
    import torch
    torch.set_num_threads(1)
    return key, sub, play_match(6, _spec(a), _spec(b), n_open, opening_plies=6, seed=seed, workers=1).summary()


def run_jobs(jobs, path, res, merge):
    """Run (key, sub, a, b, n_open, seed) jobs on 8 processes, saving after every result."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(8, mp_context=mp.get_context("spawn")) as ex:
        for i, (key, sub, r) in enumerate(ex.map(_job, jobs, chunksize=1)):
            merge(res, key, sub, r)
            if i % 8 == 0 or i == len(jobs) - 1:
                path.write_text(json.dumps(res, indent=2))
    path.write_text(json.dumps(res, indent=2))


def _load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def _merge_nested(res, key, sub, r):
    res.setdefault(key, {})[sub] = r


def stage_members():
    path = OUT / "eval_members.json"
    res = _load(path)
    jobs = [(key, name, m, ROOT / rel, 100, 4243) for key, m in members() for name, rel in REFS.items()
            if name not in res.get(key, {})]
    print("members: jobs", len(jobs), flush=True)
    run_jobs(jobs, path, res, _merge_nested)


def stage_within():
    path = OUT / "within_h2h.json"
    res = _load(path)
    jobs = []
    for run_dir in sorted(OUT.glob("F*/seed*")):
        for a, b in combinations(sorted(run_dir.glob("m*")), 2):
            key = f"{run_dir.parent.name}/{run_dir.name}/{a.name}|{b.name}"
            if key not in res:
                jobs.append((key, None, a, b, 100, 4248))
    print("within: jobs", len(jobs), flush=True)
    run_jobs(jobs, path, res, lambda r_, k, _s, r: r_.__setitem__(k, r))


def stage_ckpt():
    path = OUT / "ckpt_curves.json"
    res = _load(path)
    d3 = {"type": "minimax", "kwargs": {"depth": 3, "weights": W}}
    tmp = Path(tempfile.mkdtemp())
    jobs = []
    for key, m in members():
        cks = [(int(c.stem[1:]), c) for c in sorted((m / "ckpt").glob("g*.pt"))]
        cks.append((json.loads((m / "final.json").read_text())["games"], m / "model.pt"))
        for games, ck in cks:
            k = f"{key}/g{games:06d}"
            rd = tmp / k.replace("/", "_")
            rd.mkdir(parents=True, exist_ok=True)
            shutil.copy(m / "config.json", rd / "config.json")
            shutil.copy(ck, rd / "model.pt")
            res.setdefault(k, {"games": games})
            for sub, opp in (("minimax-d3", d3), ("ppo-s0", ROOT / REFS["ppo-s0"])):
                if sub not in res[k]:
                    jobs.append((k, sub, rd, opp, 100, 4245))
    print("ckpt: jobs", len(jobs), flush=True)
    run_jobs(jobs, path, res, _merge_nested)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["members", "within", "ckpt", "all"])
    a = p.parse_args()
    for name, fn in (("members", stage_members), ("within", stage_within), ("ckpt", stage_ckpt)):
        if a.stage in (name, "all"):
            fn()
