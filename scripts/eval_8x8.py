"""Phase 5 evaluation of the 8x8 AlphaZero model (Steps 5.4 and 5.7).

  .venv/bin/python scripts/eval_8x8.py edax  results/alphazero_8x8/seed0   # Step 5.4: vs Edax levels
  .venv/bin/python scripts/eval_8x8.py ladder results/alphazero_8x8/seed0  # Step 5.7: candidate temperatures
  .venv/bin/python scripts/eval_8x8.py verify results/alphazero_8x8/seed0  # Step 5.7: the 4 chosen levels

Levels: one model, one shared simulation count, weakened only by the move temperature
(configs/app_levels_8x8.yaml). Outputs: results/eval_8x8/<run>/{edax,ladder,levels}.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from othello.agents.registry import make_agent  # noqa: E402
from othello.env import OthelloEnv  # noqa: E402
from othello.eval.arena import make_openings, play_match  # noqa: E402
from othello.utils.config import load_json  # noqa: E402

W8 = load_json(ROOT / "results/baselines_8x8/heuristic_tuning.json")["best"]
BOTS = {"random": {"type": "random"},
        "heuristic": {"type": "heuristic", "kwargs": {"epsilon": 0.05, "weights": W8}},
        "minimax-d4": {"type": "minimax", "kwargs": {"depth": 4, "weights": W8}}}


def az(run, sims, temperature=0.0, noise=0.0):
    return {"type": "alphazero", "kwargs": {"path": str(run), "sims": sims, "temperature": temperature, "noise": noise}}


def out_dir(run):
    d = ROOT / "results/eval_8x8" / Path(run).resolve().relative_to(ROOT / "results").as_posix().replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def edax(run, levels=(1, 3, 5, 8), n_open=100, sims=150):
    res = {}
    for lv in levels:
        r = play_match(8, az(run, sims), {"type": "edax", "kwargs": {"level": lv}}, n_open, opening_plies=6,
                       seed=4249, workers=8).summary()
        res[f"edax-l{lv}"] = r
        print(f"vs edax-l{lv}: score {r['score']:.3f} +- {r['score_ci95']:.3f}", flush=True)
    for name, spec in BOTS.items():
        res[name] = play_match(8, az(run, sims), spec, n_open, opening_plies=6, seed=4249, workers=8).summary()
        print(f"vs {name}: score {res[name]['score']:.3f}", flush=True)
    (out_dir(run) / "edax.json").write_text(json.dumps({"sims": sims, "results": res}, indent=2))


def move_times(run, level, n_moves=60):
    """Mean and worst single-thread time per move at the given search settings."""
    env = OthelloEnv(8)
    agent = make_agent(az(run, level["sims"], level["temperature"], level["noise"]), env, seed=0)
    s, times, rng = env.initial_state(), [], __import__("random").Random(0)
    for op in make_openings(env, 10, 6, 77):
        s = env.initial_state()
        for m in op:
            s = env.step(s, m)
        while not env.is_terminal(s) and len(times) < n_moves:
            t = time.time()
            m = agent.act(s)
            times.append(time.time() - t)
            s = env.step(s, rng.choice(env.legal_moves(s)) if rng.random() < 0.5 else m)
        if len(times) >= n_moves:
            break
    return {"mean_s": sum(times) / len(times), "max_s": max(times)}


def ladder(run, n_open=200):
    """Candidate temperatures (shared sims) vs each other in order of strength, and vs the bots."""
    cfg = yaml.safe_load((ROOT / "configs/app_levels_8x8.yaml").read_text())
    sims, temps = cfg["sims"], sorted(cfg["candidate_temperatures"])          # T = 0 is strongest
    res = {"sims": sims, "move_time": move_times(run, {"sims": sims, "temperature": 0.0, "noise": 0.0}),
           "adjacent": {}, "vs_bots": {}}
    print("move time at", sims, "sims:", res["move_time"], flush=True)
    for strong, weak in zip(temps, temps[1:]):
        r = play_match(8, az(run, sims, strong), az(run, sims, weak), n_open, opening_plies=6, seed=4250,
                       workers=8).summary()
        res["adjacent"][f"T={strong} vs T={weak}"] = r
        print(f"T={strong} vs T={weak}: {r['score']:.3f} +- {r['score_ci95']:.3f}", flush=True)
    for t in temps:
        res["vs_bots"][f"T={t}"] = {b: play_match(8, az(run, sims, t), bs, 50, opening_plies=6, seed=4251,
                                                  workers=8).summary() for b, bs in BOTS.items()}
        print(f"T={t}", {b: round(v["score"], 3) for b, v in res["vs_bots"][f"T={t}"].items()}, flush=True)
    (out_dir(run) / "ladder.json").write_text(json.dumps(res, indent=2))


def verify(run, n_open=200):
    """The four chosen levels (configs/app_levels_8x8.yaml) against each other: each must beat the one below."""
    cfg = yaml.safe_load((ROOT / "configs/app_levels_8x8.yaml").read_text())
    sims, L = cfg["sims"], cfg["levels"]
    res = {"sims": sims, "levels": L, "adjacent": {}, "vs_bots": {}}
    for lo, hi in zip(L, L[1:]):
        r = play_match(8, az(run, sims, hi["temperature"]), az(run, sims, lo["temperature"]), n_open,
                       opening_plies=6, seed=4252, workers=8).summary()
        res["adjacent"][f"{hi['name']} vs {lo['name']}"] = r
        print(f"{hi['name']} (T={hi['temperature']}) vs {lo['name']} (T={lo['temperature']}): "
              f"{r['score']:.3f} +- {r['score_ci95']:.3f}", flush=True)
    for l in L:
        res["vs_bots"][l["name"]] = {b: play_match(8, az(run, sims, l["temperature"]), bs, 50, opening_plies=6,
                                                   seed=4253, workers=8).summary() for b, bs in BOTS.items()}
    (out_dir(run) / "levels.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["edax", "ladder", "verify"])
    p.add_argument("run")
    a = p.parse_args()
    {"edax": edax, "ladder": ladder, "verify": verify}[a.stage](Path(a.run))
