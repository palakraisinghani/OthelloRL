"""Step 2.9 evaluation: fixed-opponent DQNs vs each other, the self-play DQN and the bots.

  python scripts/eval_fixed_opponent.py panel     # Phase 1 classical DQN vs minimax-d2/d3
  python scripts/eval_fixed_opponent.py trainer   # checkpoints vs own trainer + other fixed bots
  python scripts/eval_fixed_opponent.py h2h       # round robin between the 5 learner groups
  python scripts/eval_fixed_opponent.py all

Outputs (all resumable) in results/fixed_opponent/: classical_panel.json,
trainer_curves.json, h2h.json.
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

OUT = ROOT / "results/fixed_opponent"
SEEDS = [0, 1, 2]
W = load_json(ROOT / "results/baselines/heuristic_tuning.json")["best"]
BOTS = {"random": {"type": "random"},
        "heuristic": {"type": "heuristic", "kwargs": {"epsilon": 0.05, "weights": W}},
        "minimax-d2": {"type": "minimax", "kwargs": {"depth": 2, "weights": W}},
        "minimax-d3": {"type": "minimax", "kwargs": {"depth": 3, "weights": W}}}
RUNS = {"vs-random": "random", "vs-heuristic": "heuristic", "vs-minimax-d3": "minimax-d3"}
GROUPS = {
    "dqn-random": "results/fixed_opponent/vs-random",
    "dqn-heuristic": "results/fixed_opponent/vs-heuristic",
    "dqn-minimax-d3": "results/fixed_opponent/vs-minimax-d3",
    "dqn-selfplay": "results/league/dqn/p0",
    "dqn-classical": "results/classical/dqn",
}


def match(a, b, n_open, seed):
    return play_match(6, a, b, n_open, opening_plies=6, seed=seed, workers=8).summary()


def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def panel():
    """Phase 1 classical DQN vs the bots its final.json lacks (minimax-d2/d3)."""
    path = OUT / "classical_panel.json"
    res = _load(path)
    for s in SEEDS:
        for bot in ("minimax-d2", "minimax-d3"):
            key = f"seed{s}/{bot}"
            if key not in res:
                res[key] = match({"path": str(ROOT / f"results/classical/dqn/seed{s}")}, BOTS[bot], 500, 4243)
                path.write_text(json.dumps(res, indent=2))
                print(key, round(res[key]["score"], 3), flush=True)


def trainer():
    """Each fixed-opponent run's checkpoints vs its own trainer and the other fixed bots."""
    path = OUT / "trainer_curves.json"
    res = _load(path)
    tmp = Path(tempfile.mkdtemp())
    for run in RUNS:
        for s in SEEDS:
            d = ROOT / "results/fixed_opponent" / run / f"seed{s}"
            cks = [(int(c.stem[1:]), c) for c in sorted((d / "ckpt").glob("g*.pt"))] + [(200_000, d / "model.pt")]
            for games, ck in cks:
                key = f"{run}/seed{s}/g{games:06d}"
                if key in res:
                    continue
                rd = tmp / key.replace("/", "_")      # a run-like dir so load_model works
                rd.mkdir(parents=True, exist_ok=True)
                shutil.copy(d / "config.json", rd / "config.json")
                shutil.copy(ck, rd / "model.pt")
                me = {"path": str(rd)}
                res[key] = {"games": games, "trainer": RUNS[run],
                            **{b: match(me, BOTS[b], 100, 4247) for b in ("random", "heuristic", "minimax-d3")}}
                path.write_text(json.dumps(res, indent=2))
                print(key, {b: round(res[key][b]["score"], 3) for b in ("random", "heuristic", "minimax-d3")},
                      flush=True)


def h2h():
    path = OUT / "h2h.json"
    res = _load(path)
    for ga, gb in combinations(GROUPS, 2):
        for sa in SEEDS:
            for sb in SEEDS:
                key = f"{ga}/seed{sa}|{gb}/seed{sb}"
                if key in res:
                    continue
                a = {"path": str(ROOT / GROUPS[ga] / f"seed{sa}")}
                b = {"path": str(ROOT / GROUPS[gb] / f"seed{sb}")}
                res[key] = match(a, b, 100, 4246)
                path.write_text(json.dumps(res, indent=2))
        print(ga, "vs", gb, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["panel", "trainer", "h2h", "all"])
    a = p.parse_args()
    for name, fn in (("panel", panel), ("trainer", trainer), ("h2h", h2h)):
        if a.stage in (name, "all"):
            fn()
