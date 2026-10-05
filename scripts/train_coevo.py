"""Phase 4: train co-evolving DQN populations.

  python scripts/train_coevo.py --run F1-N4 --seed 0
  python scripts/train_coevo.py --all --workers 8
  python scripts/train_coevo.py --run F1-N2 --seed 0 --games 5000 --out /tmp/x   # smoke test

Outputs results/coevo/<run>/seed<k>/m<i>/{curve.csv,final.json,config.json,model.pt,ckpt/}.
"""
import argparse
import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from othello.env import OthelloEnv  # noqa: E402
from othello.utils.config import load_config, load_json  # noqa: E402
from train_rl import resolve  # noqa: E402


def run_one(cfg, run, seed, games=None, out=None):
    import torch
    torch.set_num_threads(1)
    from othello.rl.dqn import DQNLearner
    from othello.rl.population import train_population
    weights = load_json(ROOT / "results/baselines/heuristic_tuning.json")["best"]
    cfg = resolve(copy.deepcopy(cfg), weights)
    rc = cfg["runs"][run]
    cfg["seed"] = seed
    cfg["games_per_member"] = games or rc["games_per_member"]
    if games:
        cfg["eval_every"] = max(1, games // 4)
        cfg["final_eval_openings"] = 50
    n = rc["population"]
    env = OthelloEnv(cfg["board_size"])
    learners = [DQNLearner(env, seed * 100 + i, **cfg["learner"]["kwargs"]) for i in range(n)]
    out_dir = Path(out or ROOT / cfg["out"]) / run / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"run": run, **cfg}, indent=2))
    for i in range(n):                      # member dirs look like single DQN runs (load_model works)
        (out_dir / f"m{i}").mkdir(exist_ok=True)
        member_cfg = {"run": "coevo-member", "board_size": cfg["board_size"], "population_run": run,
                      "member": i, "member_seed": seed * 100 + i,
                      "runs": {"coevo-member": {"learner": "dqn", "kwargs": cfg["learner"]["kwargs"]}}}
        (out_dir / f"m{i}" / "config.json").write_text(json.dumps(member_cfg, indent=2))
    counts = train_population(learners, cfg, out_dir)
    print("done", run, seed, "games per member:", counts, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/coevo_dqn.yaml"))
    p.add_argument("--run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int, help="override games per member (smoke tests)")
    p.add_argument("--out")
    p.add_argument("--all", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out) if a.out else ROOT / cfg["out"]
    if not a.all:
        run_one(cfg, a.run, a.seed, a.games, out)
        return
    jobs = [(r, s) for r in cfg["runs"] for s in cfg["seeds"]
            if not all((out / r / f"seed{s}" / f"m{i}" / "final.json").exists()
                       for i in range(cfg["runs"][r]["population"]))]
    # biggest first so the pool drains evenly (F1-N8 dominates)
    size = {r: cfg["runs"][r]["population"] * cfg["runs"][r]["games_per_member"] for r in cfg["runs"]}
    jobs.sort(key=lambda j: -size[j[0]])
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(job):
        r, s = job
        with open(logs / f"{r}_seed{s}.log", "w") as f:
            return r, s, subprocess.call([sys.executable, __file__, "--config", a.config, "--run", r,
                                          "--seed", str(s), "--out", str(out)], stdout=f, stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(a.workers) as ex:
        for r, s, code in ex.map(launch, jobs):
            print(f"{r} seed{s} exit={code}", flush=True)


if __name__ == "__main__":
    main()
