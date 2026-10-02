"""Environment validation & reference-opponent experiments.

  python scripts/run_baselines.py throughput   # env speed per board size
  python scripts/run_baselines.py tune         # heuristic weight grid search
  python scripts/run_baselines.py depth        # minimax depth sweep
  python scripts/run_baselines.py gate         # round-robin sanity gate + Elo
  python scripts/run_baselines.py all

Results are written as JSON to results/baselines/.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from othello.agents.registry import make_agent, spec_name  # noqa: E402
from othello.env import OthelloEnv  # noqa: E402
from othello.eval.arena import make_openings, play_game, play_match  # noqa: E402
from othello.eval.elo import fit_elo  # noqa: E402
from othello.eval.replay import save_games  # noqa: E402
from othello.utils.config import load_config, load_json, save_json  # noqa: E402

OUT = Path("results/baselines")


def throughput(cfg):
    rows = []
    for n in cfg["throughput"]["sizes"]:
        r = play_match(n, {"type": "random"}, {"type": "random"}, cfg["throughput"]["openings"],
                       cfg["opening_plies"], cfg["seed"], workers=1, keep_games=True)
        plies = sum(len(g["moves"]) for g in r.games)
        rows.append({"board_size": n, "games": r.n, "games_per_sec": r.n / r.seconds,
                     "plies_per_sec": plies / r.seconds, "mean_plies_per_game": plies / r.n,
                     "black_score_random": sum((g["black_minus_white"] > 0) + 0.5 * (g["black_minus_white"] == 0)
                                                for g in r.games) / r.n})
        print(rows[-1])
    save_json(OUT / "throughput.json", rows)


def tune(cfg):
    t = cfg["heuristic_tuning"]
    keys = list(t["grid"])
    rows = []
    for vals in itertools.product(*(t["grid"][k] for k in keys)):
        w = dict(zip(keys, vals))
        cand = {"type": "heuristic", "kwargs": {"weights": w}}
        scores = {}
        for opp in t["opponents"]:
            r = play_match(t["board_size"], cand, opp, t["openings"], cfg["opening_plies"],
                           cfg["seed"], cfg["workers"])
            scores[opp["name"]] = r.score
        rows.append({"weights": w, "scores": scores, "mean_score": sum(scores.values()) / len(scores)})
        print(w, {k: round(v, 3) for k, v in scores.items()})
    rows.sort(key=lambda r: -r["mean_score"])
    save_json(OUT / "heuristic_tuning.json", {"best": rows[0]["weights"], "rows": rows})


def best_weights():
    return load_json(OUT / "heuristic_tuning.json")["best"]


def depth_sweep(cfg):
    d = cfg["minimax_depth_sweep"]
    n, w = d["board_size"], best_weights()
    eps = cfg["gate"]["heuristic_epsilon"]
    heur = {"type": "heuristic", "kwargs": {"weights": w, "epsilon": eps}}
    rows = []
    for depth in d["depths"]:
        mm = {"type": "minimax", "kwargs": {"depth": depth, "weights": w}}
        r = play_match(n, mm, heur, d["openings"], cfg["opening_plies"], cfg["seed"], cfg["workers"])
        # search cost: nodes and seconds per move, measured in-process on a few games
        env = OthelloEnv(n)
        agent = make_agent(mm, env, seed=1)
        opp = make_agent(heur, env, seed=2)
        moves = 0
        t0 = time.time()
        for i, op in enumerate(make_openings(env, 10, cfg["opening_plies"], 123)):
            black, white = (agent, opp) if i % 2 == 0 else (opp, agent)
            _, ms = play_game(env, black, white, op)
            moves += len(ms) // 2
        secs = time.time() - t0
        rows.append({"depth": depth, **r.summary(), "nodes_per_move": agent.nodes / moves,
                     "sec_per_game_both_sides": secs / 10})
        print(depth, round(r.score, 3), round(agent.nodes / moves))
    save_json(OUT / "minimax_depth.json", rows)


def gate(cfg):
    g = cfg["gate"]
    w = best_weights()
    specs = [
        {"name": "random", "type": "random"},
        {"name": "heuristic", "type": "heuristic", "kwargs": {"weights": w, "epsilon": g["heuristic_epsilon"]}},
        {"name": f"minimax-d{g['minimax_depth']}", "type": "minimax",
         "kwargs": {"depth": g["minimax_depth"], "weights": w}},
    ]
    out = {"weights": w, "sizes": {}}
    for n, n_open in g["sizes"].items():
        n = int(n)
        matches, elo_in = [], {}
        for a, b in itertools.combinations(reversed(specs), 2):   # stronger agent as A
            r = play_match(n, a, b, n_open, cfg["opening_plies"], cfg["seed"], cfg["workers"], keep_games=True)
            matches.append({"a": spec_name(a), "b": spec_name(b), **r.summary()})
            elo_in[(spec_name(a), spec_name(b))] = (r.wins + 0.5 * r.draws, r.n)
            save_games(OUT / f"replays/{n}x{n}_{spec_name(a)}_vs_{spec_name(b)}.jsonl", r.games,
                       {"board_size": n, "a": spec_name(a), "b": spec_name(b)})
            print(n, matches[-1])
        out["sizes"][n] = {"matches": matches, "elo": fit_elo(elo_in, anchor="random")}
    save_json(OUT / "gate.json", out)


def diag(cfg):
    """Why does the 1-ply heuristic miss the >=95% bar vs random on 6x6?"""
    w, n_open = best_weights(), cfg["gate"]["sizes"][6]
    rows = []
    for label, spec in [
        ("heuristic eps=0.05", {"type": "heuristic", "kwargs": {"weights": w, "epsilon": cfg["gate"]["heuristic_epsilon"]}}),
        ("heuristic eps=0", {"type": "heuristic", "kwargs": {"weights": w}}),
        ("minimax-d1", {"type": "minimax", "kwargs": {"depth": 1, "weights": w}}),
        ("minimax-d2", {"type": "minimax", "kwargs": {"depth": 2, "weights": w}}),
    ]:
        for size in (6, 8):
            r = play_match(size, spec, {"type": "random"}, n_open if size == 6 else 200,
                           cfg["opening_plies"], cfg["seed"], cfg["workers"])
            rows.append({"agent": label, "board_size": size, **r.summary()})
            print(rows[-1])
    save_json(OUT / "gate_diag.json", rows)


def openings(cfg):
    """Does opening length explain the heuristic-vs-random shortfall on 6x6?"""
    w = best_weights()
    rows = []
    for eps in (cfg["gate"]["heuristic_epsilon"], 0.0):
        for plies in (0, 2, 4, 6):
            h = {"type": "heuristic", "kwargs": {"weights": w, "epsilon": eps}}
            r = play_match(6, h, {"type": "random"}, cfg["gate"]["sizes"][6], plies, cfg["seed"], cfg["workers"])
            distinct = len({tuple(o) for o in make_openings(OthelloEnv(6), cfg["gate"]["sizes"][6], plies, cfg["seed"])})
            rows.append({"epsilon": eps, "opening_plies": plies, "distinct_openings": distinct, **r.summary()})
            print(rows[-1])
    save_json(OUT / "gate_openings.json", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["throughput", "tune", "depth", "gate", "diag", "openings", "all"])
    p.add_argument("--config", default="configs/baselines.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    stages = {"throughput": throughput, "tune": tune, "depth": depth_sweep, "gate": gate, "diag": diag, "openings": openings}
    for name, fn in stages.items():
        if a.stage in (name, "all"):
            fn(cfg)
