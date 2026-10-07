"""AlphaZero training loop: generate self-play games in worker processes, train on a replay window.

Each iteration: the current network plays `games_per_iter` self-play games
(MCTS with Dirichlet noise at the root, visit-count sampling for the first
`temp_plies` plies), the examples enter a sliding replay window, and the
network takes `train_steps` gradient steps on the loss
    (z - v)^2  -  pi . log p  (+ weight decay),
with each batch randomly transformed by one of the 8 board symmetries.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from othello.agents.registry import make_agent
from othello.env import OthelloEnv
from othello.eval.arena import make_openings
from othello.rl.alphazero.net import count_params
from othello.rl.alphazero.selfplay import PolicyOnlyAgent, az_evaluate, build_net, worker
from othello.rl.common import evaluate as policy_evaluate
from othello.symmetry import get_symmetry


class Augmenter:
    """Applies one random dihedral transform per sample to planes and policy targets."""

    def __init__(self, n, rng):
        sym = get_symmetry(n)
        self.n, self.rng = n, rng
        nsq = n * n
        # gather indices: transformed[..., j] = original[..., inv[t][j]]
        self.inv = np.array([[sym.inv[t][j] for j in range(nsq)] for t in range(8)])
        self.pi_idx = np.concatenate([self.inv, np.full((8, 1), nsq)], axis=1)

    def __call__(self, x, pi):
        B, C = x.shape[:2]
        t = self.rng.integers(0, 8, B)
        xf = x.reshape(B, C, -1)
        x2 = np.take_along_axis(xf, self.inv[t][:, None, :], axis=2).reshape(x.shape)
        pi2 = np.take_along_axis(pi, self.pi_idx[t], axis=1)
        return x2, pi2


class _NetLearner:
    """Adapter so the shared `evaluate` can score the raw policy (no search)."""

    def __init__(self, env, net):
        self.env = env
        self.agent = PolicyOnlyAgent(env, net)

    def act(self, state, explore=False):
        return self.agent.act(state)

    def probe(self, state):
        return None


def train_alphazero(cfg: dict, out_dir: Path, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    env = OthelloEnv(cfg["board_size"])
    net = build_net(env.n, cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    aug = Augmenter(env.n, rng)
    torch.set_num_threads(cfg.get("train_threads", 4))

    train_openings = make_openings(env, 20_000, cfg["opening_plies"], 1_000 + seed)
    eval_openings = make_openings(env, cfg["eval_openings"], cfg["opening_plies"], cfg["eval_seed"])
    eval_opp = make_agent(cfg["opponent"], env, seed=10_000 + seed)

    buf = deque(maxlen=cfg["buffer_positions"])
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(cfg["workers"])
    games = positions = it = 0
    next_eval = 0
    t0 = time.time()
    losses = []
    log = open(out_dir / "curve.csv", "w", newline="")
    writer = None
    meta = {"params": count_params(net)}

    def do_eval():
        nonlocal writer
        net.eval()
        r = az_evaluate(net, env, eval_opp, eval_openings, cfg["eval_sims"], seed=seed)
        rp = policy_evaluate(_NetLearner(env, net), eval_opp, eval_openings)
        lp = np.mean(losses, axis=0) if losses else (float("nan"), float("nan"))
        losses.clear()
        row = {"games": games, "seconds": time.time() - t0, "win_rate": r["win_rate"], "score": r["score"],
               "draw_rate": r["draw_rate"], "margin": r["margin"], "policy_score": rp["score"],
               "loss_pi": lp[0], "loss_v": lp[1], "positions": positions, "buffer": len(buf), "iter": it}
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        log.flush()
        print("alphazero", seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()},
              flush=True)

    try:
        while True:
            if games >= next_eval:
                do_eval()
                next_eval += cfg["eval_every"]
            if games >= cfg["train_games"]:
                break
            # ---- self-play
            net.eval()
            sd = {k: v.clone() for k, v in net.state_dict().items()}
            n_games = cfg["games_per_iter"]
            ops = []
            for _ in range(n_games):
                ops.append([] if prng.random() < cfg["start_pos_frac"] else prng.choice(train_openings))
            per = [ops[w::cfg["workers"]] for w in range(cfg["workers"])]
            args = [(sd, cfg, int(rng.integers(1 << 31)), p) for p in per if p]
            for ex in pool.map(worker, args):
                buf.extend(ex)
                positions += len(ex)
            prev = games
            games += n_games
            it += 1
            ck = cfg.get("checkpoint_every")
            if ck and games // ck > prev // ck:
                (out_dir / "ckpt").mkdir(exist_ok=True)
                torch.save(net.state_dict(), out_dir / "ckpt" / f"g{games:06d}.pt")
            # ---- training
            net.train()
            steps = cfg["train_steps"] if len(buf) >= cfg["min_buffer"] else 0
            for _ in range(steps):
                idx = rng.integers(0, len(buf), cfg["batch"])
                x = np.stack([buf[i][0] for i in idx]).astype(np.float32)
                pi = np.stack([buf[i][1] for i in idx])
                z = np.array([buf[i][2] for i in idx], np.float32)
                x, pi = aug(x, pi)
                logits, v = net(torch.from_numpy(x))
                # illegal actions have pi = 0 and are excluded via the legal plane mask
                legal = torch.from_numpy(np.concatenate(
                    [x[:, 2].reshape(len(idx), -1), (x[:, 2].reshape(len(idx), -1).sum(1, keepdims=True) == 0)],
                    axis=1).astype(bool))
                logp = F.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)
                loss_pi = -(torch.from_numpy(pi) * logp).sum(1).mean()
                loss_v = F.mse_loss(v, torch.from_numpy(z))
                loss = loss_pi + loss_v
                opt.zero_grad()
                loss.backward()
                opt.step()
                # keep weights out of the denormal range (flush_denormal may not reach every intra-op thread)
                with torch.no_grad():
                    for prm in net.parameters():
                        prm.masked_fill_(prm.abs() < 1e-30, 0.0)
                losses.append((loss_pi.item(), loss_v.item()))
        net.eval()
        torch.save(net.state_dict(), out_dir / "model.pt")
        meta.update({"games": games, "positions": positions, "train_seconds": time.time() - t0})
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    finally:
        pool.close()
        pool.join()
        log.close()
    return net


def final_eval_az(net, cfg, out_dir: Path, seed: int):
    env = OthelloEnv(cfg["board_size"])
    openings = make_openings(env, cfg["final_eval_openings"], cfg["opening_plies"], cfg["eval_seed"] + 1)
    res = {}
    for name, spec in cfg["final_opponents"].items():
        ops = openings if not name.startswith("minimax") else openings[: len(openings) // 2]
        opp = make_agent(spec, env, seed=20_000 + seed)
        res[name] = az_evaluate(net, env, opp, ops, cfg["eval_sims"], seed=seed)
        rp = policy_evaluate(_NetLearner(env, net), make_agent(spec, env, seed=20_000 + seed), ops)
        res[f"policy_only/{name}"] = {**rp, "games": 2 * len(ops)}
        print("final", name, res[name], "policy-only", rp, flush=True)
    meta = json.loads((out_dir / "meta.json").read_text())
    res["train_seconds"] = meta["train_seconds"]
    (out_dir / "final.json").write_text(json.dumps(res, indent=2))
    return res
