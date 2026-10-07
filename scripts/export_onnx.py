"""Export a trained AlphaZero model for the Othello-Player app (PLAN Step 5.6).

  .venv/bin/python scripts/export_onnx.py results/alphazero_8x8/seed0 --out results/export/8x8_seed0

Writes:
  oroletto-8x8.onnx   network: planes [batch, 3, n, n] -> policy_logits [batch, n*n+1], value [batch]
  oroletto-8x8.json   manifest: layout, search settings, levels (filled by Step 5.7), provenance, SHA-256
  parity_8x8.json     fixtures for the app's parity tests: network outputs and exact MCTS visit counts

Input planes, from the side to move: [own discs, opponent discs, legal moves]; square index
row*n + col with a1 = 0 at the top-left; the last policy index is pass. Search without root
noise is deterministic (ties in selection go to the first move in legal-move order), so visit
counts must match exactly; only the final choice among equally visited moves is random.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.env import OthelloEnv  # noqa: E402
from othello.rl.alphazero.mcts import MCTS, Evaluator, Node  # noqa: E402
from othello.rl.alphazero.net import FastEncoder  # noqa: E402
from othello.rl.alphazero.selfplay import build_net  # noqa: E402

PLACEHOLDER_LEVELS = [  # SPEC section 3.3 starting values; replaced by Step 5.7's calibration
    {"level": 1, "name": "Easy", "sims": 10, "temperature": 1.0, "noise": 0.3},
    {"level": 2, "name": "Medium", "sims": 50, "temperature": 0.5, "noise": 0.0},
    {"level": 3, "name": "Hard", "sims": 200, "temperature": 0.1, "noise": 0.0},
    {"level": 4, "name": "Master", "sims": 800, "temperature": 0.0, "noise": 0.0},
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_positions(env, n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = env.initial_state()
        stop = rng.randrange(0, 50)
        for _ in range(stop):
            if env.is_terminal(s):
                break
            s = env.step(s, rng.choice(env.legal_moves(s)))
        if not env.is_terminal(s) and len(env.legal_moves(s)) > 1:
            out.append(s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=200)
    p.add_argument("--parity-sims", type=int, nargs="*", default=[16, 64, 200])
    a = p.parse_args()
    run = Path(a.run_dir).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run / "config.json").read_text())
    n = cfg["board_size"]
    env = OthelloEnv(n)
    net = build_net(n, cfg)
    net.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    net.eval()
    stem = f"oroletto-{n}x{n}"

    # ---- ONNX export
    onnx_path = out / f"{stem}.onnx"
    dummy = torch.zeros(1, 3, n, n)
    torch.onnx.export(net, dummy, onnx_path, input_names=["planes"], output_names=["policy_logits", "value"],
                      dynamic_axes={"planes": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False)

    # ---- verify against PyTorch on real positions
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    enc = FastEncoder(env)
    positions = sample_positions(env, a.positions, seed=7171)
    x = np.stack([enc.encode(s) for s in positions]).astype(np.float32)
    with torch.no_grad():
        tp, tv = net(torch.from_numpy(x))
    op, ov = sess.run(None, {"planes": x})
    max_diff = float(max(np.abs(op - tp.numpy()).max(), np.abs(ov - tv.numpy()).max()))
    print(f"ONNX vs PyTorch: max abs difference {max_diff:.2e} over {len(positions)} positions")
    assert max_diff < 1e-4, "ONNX export does not match PyTorch"

    # ---- parity fixtures: network outputs + exact visit counts at fixed simulation counts
    ev = Evaluator(env, net)
    fixtures = []
    for s in positions[:50]:
        legal = env.legal_moves(s)
        item = {"black": s.black, "white": s.white, "to_move": "black" if s.player == 0 else "white",
                "planes": enc.encode(s).astype(int).tolist(), "legal_moves": legal}
        with torch.no_grad():
            pl, v = net(torch.from_numpy(enc.encode(s)[None]))
        item["policy_logits"] = [round(float(z), 6) for z in pl[0]]
        item["value"] = round(float(v[0]), 6)
        item["search"] = {}
        for sims in a.parity_sims:
            mcts = MCTS(env, ev, c_puct=cfg["c_puct"], rng=np.random.default_rng(0))
            root = mcts.search([Node(s)], sims, add_noise=False)[0]
            best = max(root.N)
            item["search"][str(sims)] = {"visits": dict(zip(map(str, root.moves), root.N)),
                                         "most_visited": [m for m, c in zip(root.moves, root.N) if c == best]}
        fixtures.append(item)
    (out / f"parity_{n}x{n}.json").write_text(json.dumps(
        {"board_size": n, "c_puct": cfg["c_puct"], "note": "visit counts are exact; ties broken randomly only "
         "when choosing among most_visited", "positions": fixtures}, indent=1))

    # ---- manifest
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    final = json.loads((run / "final.json").read_text()) if (run / "final.json").exists() else {}
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    manifest = {
        "model": onnx_path.name, "sha256": sha256(onnx_path), "board_size": n,
        "input": {"name": "planes", "shape": ["batch", 3, n, n], "dtype": "float32",
                  "planes": ["own discs", "opponent discs", "legal moves"], "perspective": "side to move",
                  "square_index": "row * n + col, a1 = 0 top-left", "pass_index": n * n},
        "outputs": {"policy_logits": ["batch", n * n + 1], "value": ["batch"]},
        "search": {"algorithm": "PUCT MCTS (OthelloRL othello/rl/alphazero/mcts.py)", "c_puct": cfg["c_puct"],
                   "dirichlet_alpha": cfg["dirichlet_alpha"], "move_choice": "sample by N^(1/T); T=0 most-visited",
                   "value_sign": "negated each ply (every action, including pass, flips the side to move)"},
        "levels": PLACEHOLDER_LEVELS, "levels_calibrated": False,
        "network": {"blocks": cfg["blocks"], "channels": cfg["channels"],
                    "parameters": meta.get("params"), "training_games": meta.get("games")},
        "provenance": {"othellorl_commit": commit, "run": str(run.relative_to(ROOT)),
                       "final_scores": {k: v.get("score") for k, v in final.items() if isinstance(v, dict)}},
        "onnx_check": {"positions": len(positions), "max_abs_diff_vs_pytorch": max_diff},
    }
    (out / f"{stem}.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
