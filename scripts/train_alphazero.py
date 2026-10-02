"""Train AlphaZero-style agents by self-play.

  python scripts/train_alphazero.py --seed 0                    # one full run
  python scripts/train_alphazero.py --all                       # every seed, one after another
  python scripts/train_alphazero.py --seed 0 --games 2048 --workers 2 --out /tmp/x   # smoke test

Outputs results/alphazero/seed<k>/{curve.csv,final.json,config.json,meta.json,model.pt}.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.rl.alphazero.train import final_eval_az, train_alphazero  # noqa: E402
from othello.utils.config import load_config, load_json  # noqa: E402
from train_rl import resolve  # noqa: E402


def run(cfg, seed, out_root: Path):
    out_dir = out_root / f"seed{seed}"
    if (out_dir / "final.json").exists():
        print("skip, done:", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"run": "alphazero", "seed": seed, **cfg}, indent=2))
    net = train_alphazero(cfg, out_dir, seed)
    final_eval_az(net, cfg, out_dir, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/alphazero.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true")
    p.add_argument("--games", type=int, help="override train_games (smoke tests)")
    p.add_argument("--workers", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    weights = load_json(ROOT / "results/baselines/heuristic_tuning.json")["best"]
    cfg = resolve(copy.deepcopy(load_config(a.config)), weights)
    if a.games:
        cfg["train_games"] = a.games
        cfg["eval_every"] = max(cfg["games_per_iter"], a.games // 4)
        cfg["final_eval_openings"] = 50
        cfg["min_buffer"] = 1000
    if a.workers:
        cfg["workers"] = a.workers
    out_root = Path(a.out) if a.out else ROOT / cfg["out"]
    for seed in (cfg["seeds"] if a.all else [a.seed]):
        run(cfg, seed, out_root)


if __name__ == "__main__":
    main()
