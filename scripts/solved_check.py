"""Compare a trained AlphaZero model with the known solution of 6x6 Othello.

Perfect play from the standard start ends 16-20 (White wins by 4; Feinstein 1993).
We (1) read the value head's prediction for the start position and (2) let the
model play itself from the start with greedy MCTS at several search budgets.

  python scripts/solved_check.py results/alphazero/seed1
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.env import OthelloEnv  # noqa: E402
from othello.rl.alphazero.mcts import Evaluator  # noqa: E402
from othello.rl.alphazero.selfplay import AZAgent, build_net  # noqa: E402


def main(run_dir, budgets=(64, 200, 800), seeds=(0, 1, 2)):
    d = Path(run_dir).resolve()
    cfg = json.loads((d / "config.json").read_text())
    env = OthelloEnv(cfg["board_size"])
    net = build_net(env.n, cfg)
    net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    net.eval()
    s0 = env.initial_state()
    _, v = Evaluator(env, net)([s0])
    out = {"model": str(d.relative_to(ROOT)), "perfect_play": {"black": 16, "white": 20},
           "start_value_black": v[0], "games": []}
    print("value head at start (Black's view):", round(v[0], 3))
    for sims in budgets:
        for seed in seeds:
            black = AZAgent(env, net, sims=sims, seed=seed)
            white = AZAgent(env, net, sims=sims, seed=seed + 100)
            s, moves = s0, []
            while not env.is_terminal(s):
                m = (black if s.player == 0 else white).act(s)
                moves.append(env.action_name(m))
                s = env.step(s, m)
            b, w = env.counts(s)
            out["games"].append({"sims": sims, "seed": seed, "black": b, "white": w, "moves": moves})
            print(sims, seed, f"{b}-{w}", " ".join(moves))
    (d.parent / "solved_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
