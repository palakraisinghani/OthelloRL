"""Train RL learners: classical (vs the fixed heuristic bot) or self-play (opponent pool).

  python scripts/train_rl.py --run q-learning --seed 0      # one run
  python scripts/train_rl.py --all --workers 8               # every run x seed
  python scripts/train_rl.py --run dqn --seed 0 --games 2000 # quick smoke test

Outputs results/classical/<run>/seed<k>/{curve.csv,final.json,config.json}.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.env import OthelloEnv  # noqa: E402
from othello.utils.config import load_config, load_json  # noqa: E402

OUT = ROOT / "results" / "classical"


def resolve(obj, weights):
    if isinstance(obj, dict):
        return {k: resolve(v, weights) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, weights) for v in obj]
    return weights if obj == "TUNED" else obj


def build(run_cfg, env, seed):
    kind, kw = run_cfg["learner"], dict(run_cfg.get("kwargs", {}))
    if kind == "tabular":
        from othello.rl.tabular import TabularLearner
        return TabularLearner(env, seed, **kw)
    if kind == "linear":
        from othello.rl.linear import LinearQLearner
        return LinearQLearner(env, seed, **kw)
    if kind == "dqn":
        import torch
        torch.set_num_threads(1)
        from othello.rl.dqn import DQNLearner
        return DQNLearner(env, seed, **kw)
    if kind == "ppo":
        import torch
        torch.set_num_threads(1)
        from othello.rl.ppo import PPOLearner
        return PPOLearner(env, seed, **kw)
    raise ValueError(kind)


def run_one(cfg, run, seed, games=None, out=OUT):
    from othello.rl.common import final_eval, train
    from othello.rl.linear import disc_margin_reward
    weights = load_json(ROOT / "results/baselines/heuristic_tuning.json")["best"]
    cfg = resolve(copy.deepcopy(cfg), weights)
    cfg["seed"] = seed
    if games:
        cfg["train_games"] = games
        cfg["eval_every"] = max(1, games // 4)
    run_cfg = cfg["runs"][run]
    env = OthelloEnv(cfg["board_size"])
    learner = build(run_cfg, env, seed)
    learner.name = run
    out_dir = out / run / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"run": run, **cfg}, indent=2))
    reward_fn = disc_margin_reward if run_cfg.get("reward") == "disc_margin" else None
    t0 = time.time()
    pool = None
    if "selfplay" in run_cfg:
        from othello.rl.pool import OpponentPool
        pool = OpponentPool(learner, seed=seed + 31, **run_cfg["selfplay"])
    train(learner, cfg, out_dir, reward_fn, pool)
    res = final_eval(learner, cfg, out_dir, cfg["final_opponents"])
    res["train_seconds"] = time.time() - t0
    (out_dir / "final.json").write_text(json.dumps(res, indent=2))
    if hasattr(learner, "save"):
        learner.save(out_dir / "model.pt")
    if hasattr(learner, "w"):
        (out_dir / "weights.json").write_text(json.dumps(dict(zip(learner.features, learner.w.tolist()))))
    print("done", run, seed, json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/classical.yaml"))
    p.add_argument("--run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int)
    p.add_argument("--out", help="output root (default: config 'out', else results/classical)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--only", nargs="*", help="with --all: subset of runs")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    cfg = load_config(a.config)
    a.out = a.out or str(ROOT / cfg.get("out", "results/classical"))
    if not a.all:
        run_one(cfg, a.run, a.seed, a.games, Path(a.out))
        return
    jobs = [(r, s) for r in (a.only or cfg["runs"]) for s in cfg["seeds"]
            if not (Path(a.out) / r / f"seed{s}" / "final.json").exists()]
    # slowest runs first so the pool drains evenly
    order = {"dqn": 0, "ppo": 0, "linear": 1}
    jobs.sort(key=lambda j: order.get(j[0].split("-")[0], 2))
    logs = Path(a.out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(job):
        r, s = job
        with open(logs / f"{r}_seed{s}.log", "w") as f:
            cmd = [sys.executable, __file__, "--config", a.config, "--run", r, "--seed", str(s), "--out", a.out]
            if a.games:
                cmd += ["--games", str(a.games)]
            return r, s, subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(a.workers) as ex:
        for r, s, code in ex.map(launch, jobs):
            print(f"{r} seed{s} exit={code}", flush=True)


if __name__ == "__main__":
    main()
