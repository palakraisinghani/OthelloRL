"""Worked example of one AlphaZero search for the report (Sec. "How AlphaZero works").

  .venv/bin/python scripts/make_az_example.py

Runs Oroletto's search (no root noise, 64 simulations as in 6x6 self-play) on sampled
positions and keeps one where the move with the highest mean value Q is *not* the
most-visited move, then re-searches it at larger budgets.

Writes results/az_example.json, report/figures/az_example_board.tex (TikZ) and
report/tables/az_example.tex.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.env import OthelloEnv  # noqa: E402
from scripts.make_rule_diagrams import tikz  # noqa: E402
from othello.rl.alphazero.mcts import MCTS, Evaluator, Node  # noqa: E402
from othello.rl.alphazero.selfplay import build_net  # noqa: E402

SIMS = 64
BUDGETS = (64, 800, 3200)


def write_assets(env, s, moves, rows):
    (ROOT / "report/figures/az_example_board.tex").write_text(tikz(env, s, ghosts=moves, scale=0.62) + "\n")
    lines = []
    for r in rows:
        cells = [r["move"], f"{r['prior']:.2f}"]
        for b in BUDGETS:
            col = [x["search"][str(b)] for x in rows]
            top_n = max(c["visits"] for c in col)
            top_q = max(c["q"] for c in col if c["q"] is not None)
            c = r["search"][str(b)]
            n = rf"\textbf{{{c['visits']}}}" if c["visits"] == top_n else str(c["visits"])
            if c["q"] is None:
                q = "--"
            else:
                q = f"{c['q']:+.2f}"
                q = rf"\underline{{{q}}}" if c["q"] == top_q else q
            cells += [n, q]
        lines.append(" & ".join(cells) + r" \\")
    (ROOT / "report/tables/az_example.tex").write_text("\n".join(lines) + "\n")


def main():
    ptr = json.loads((ROOT / "models/oroletto.json").read_text())
    run = ROOT / ptr["path"]
    cfg = json.loads((run / "config.json").read_text())
    env = OthelloEnv(cfg["board_size"])
    net = build_net(env.n, cfg)
    net.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    net.eval()
    ev = Evaluator(env, net)
    rng = random.Random(3)
    for _ in range(2000):
        s = env.initial_state()
        for _ in range(rng.randrange(6, 16)):
            if env.is_terminal(s):
                break
            s = env.step(s, rng.choice(env.legal_moves(s)))
        if env.is_terminal(s) or not 4 <= len(env.legal_moves(s)) <= 7:
            continue
        root = MCTS(env, ev, c_puct=cfg["c_puct"], rng=np.random.default_rng(0)).search([Node(s)], SIMS, False)[0]
        q = [w / n if n else None for w, n in zip(root.W, root.N)]
        best_n = max(range(len(root.N)), key=lambda i: root.N[i])
        best_q = max((i for i in range(len(q)) if q[i] is not None), key=lambda i: q[i])
        if best_q != best_n and root.N[best_q] <= 3 and q[best_q] > q[best_n] + 0.05:
            search = {}
            for b in BUDGETS:
                r = MCTS(env, ev, c_puct=cfg["c_puct"], rng=np.random.default_rng(0)).search([Node(s)], b, False)[0]
                search[b] = {env.action_name(m): (n, w / n if n else None) for m, n, w in zip(r.moves, r.N, r.W)}
            rows = [{"move": env.action_name(m), "prior": p,
                     "search": {str(b): {"visits": search[b][env.action_name(m)][0],
                                         "q": search[b][env.action_name(m)][1]} for b in BUDGETS}}
                    for m, p in zip(root.moves, root.P)]
            rows.sort(key=lambda r: -r["prior"])
            out = {"sims": SIMS, "budgets": BUDGETS, "c_puct": cfg["c_puct"], "model": ptr["path"],
                   "board": env.render(s), "to_move": "black" if s.player == 0 else "white",
                   "black": s.black, "white": s.white, "rows": rows}
            (ROOT / "results/az_example.json").write_text(json.dumps(out, indent=1))
            write_assets(env, s, root.moves, rows)
            print(env.render(s))
            for r in rows:
                print(r)
            return
    print("no example found")


if __name__ == "__main__":
    main()
