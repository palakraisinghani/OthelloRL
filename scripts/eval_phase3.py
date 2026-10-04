"""Phase 3 evaluation: held-out panel calibration and evaluation of the league DQN runs.

  python scripts/eval_phase3.py calibrate     # Step 3.3: DQN+ vs minimax-d2/d3
  python scripts/eval_phase3.py evaluate      # Step 3.5: all models vs panel + in-league bots
  python scripts/eval_phase3.py h2h           # Step 3.5: mix x mix head-to-head matrix
  python scripts/eval_phase3.py ckpt          # Step 3.5: checkpoint curves (H4)
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.eval.arena import play_match  # noqa: E402
from othello.utils.config import load_json  # noqa: E402

OUT = ROOT / "results" / "league"
MIXES = ["p0", "p25", "p50", "p75", "p100"]
SEEDS = [0, 1, 2]
W = load_json(ROOT / "results/baselines/heuristic_tuning.json")["best"]


def mm(d):
    return {"type": "minimax", "kwargs": {"depth": d, "weights": W}}


HELD_OUT_BOTS = {"minimax-d2": mm(2), "minimax-d3": mm(3)}
IN_LEAGUE = {"random": {"type": "random"},
             "heuristic": {"type": "heuristic", "kwargs": {"epsilon": 0.05, "weights": W}},
             "minimax-d4": mm(4)}
PPO = {f"ppo-s{k}": {"path": str(ROOT / f"results/selfplay/ppo/seed{k}")} for k in SEEDS}
DQNP = {f"dqn+-s{k}": {"path": str(ROOT / f"results/selfplay/dqn-plus/seed{k}")} for k in SEEDS}


def model(mix, seed):
    return {"path": str(ROOT / f"results/league/dqn/{mix}/seed{seed}")}


def match(a, b, n_open, seed, workers=8):
    r = play_match(6, a, b, n_open, opening_plies=6, seed=seed, workers=workers)
    return r.summary()


def calibrate(a):
    out = {}
    for name, bot in HELD_OUT_BOTS.items():
        rows = [match(spec, bot, 200, 5151) for spec in DQNP.values()]
        out[name] = {"per_seed": rows, "mean_score": sum(r["score"] for r in rows) / len(rows)}
        print(name, round(out[name]["mean_score"], 3), [round(r["score"], 3) for r in rows], flush=True)
    out["accepted"] = all(0.2 <= out[n]["mean_score"] <= 0.8 for n in HELD_OUT_BOTS)
    (OUT / "panel_calibration.json").write_text(json.dumps(out, indent=2))
    print("accepted:", out["accepted"])


def evaluate(a):
    path = OUT / "dqn_eval.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for mix in MIXES:
        for s in SEEDS:
            key = f"{mix}/seed{s}"
            if key in res:
                continue
            me = model(mix, s)
            r = {}
            for k in SEEDS:                                   # vs self-play (p0) models
                if not (mix == "p0" and k == s):
                    r[f"p0-s{k}"] = match(me, model("p0", k), 100, 4243)
            for name, bot in HELD_OUT_BOTS.items():
                r[name] = match(me, bot, 500, 4243)
            for name, spec in {**PPO, **DQNP}.items():
                r[name] = match(me, spec, 100, 4243)
            for name, bot in IN_LEAGUE.items():
                r[name] = match(me, bot, 500, 4243)
            res[key] = r
            path.write_text(json.dumps(res, indent=2))
            print(key, {n: round(v["score"], 3) for n, v in r.items()}, flush=True)


def h2h(a):
    path = OUT / "dqn_h2h_matrix.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for i, ma in enumerate(MIXES):
        for mb in MIXES[i + 1:]:
            for sa in SEEDS:
                for sb in SEEDS:
                    key = f"{ma}/seed{sa}|{mb}/seed{sb}"
                    if key not in res:
                        res[key] = match(model(ma, sa), model(mb, sb), 100, 4244)
                        path.write_text(json.dumps(res, indent=2))
            print(ma, mb, flush=True)


def ckpt(a):
    """Checkpoint curves: each saved checkpoint vs minimax-d3 and Phase 2 PPO seed 0."""
    import shutil
    import tempfile
    path = OUT / "dqn_ckpt_curves.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    tmp = Path(tempfile.mkdtemp())
    for mix in MIXES:
        for s in SEEDS:
            run = ROOT / f"results/league/dqn/{mix}/seed{s}"
            cks = sorted((run / "ckpt").glob("g*.pt"))
            final = run / "model.pt"                 # the end-of-training model is the last curve point
            for ck in cks + [final]:
                stem = ck.stem if ck != final else "g200000"
                key = f"{mix}/seed{s}/{stem}"
                if key in res:
                    continue
                d = tmp / key.replace("/", "_")      # a run-like dir so load_model works
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(run / "config.json", d / "config.json")
                shutil.copy(ck, d / "model.pt")
                me = {"path": str(d)}
                res[key] = {"games": int(stem[1:]),
                            "minimax-d3": match(me, HELD_OUT_BOTS["minimax-d3"], 100, 4245),
                            "ppo-s0": match(me, PPO["ppo-s0"], 100, 4245)}
                path.write_text(json.dumps(res, indent=2))
                print(key, {k: round(v["score"], 3) for k, v in res[key].items() if k != "games"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["calibrate", "evaluate", "h2h", "ckpt"])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"calibrate": calibrate, "evaluate": evaluate, "h2h": h2h, "ckpt": ckpt}[a.stage](a)
