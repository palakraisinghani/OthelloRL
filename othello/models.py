"""Load trained models from a run directory (or file) as playable agents."""
from __future__ import annotations

import json
from pathlib import Path

from othello.agents.base import Agent


class LearnerAgent(Agent):
    """Greedy (no exploration) wrapper around a trained learner."""

    def __init__(self, learner, name):
        super().__init__(learner.env)
        self.learner, self.name = learner, name

    def act(self, state):
        return self.learner.act(state, explore=False)


def load_model(path, env, sims=None) -> Agent:
    """`path` is a run directory (results/classical/<run>/seed<k>) or a file in it.

    Supported: AlphaZero, PPO and DQN (`model.pt` + `config.json`), linear FA
    (`weights.json`) and tabular learners (`table.pkl.gz`). `sims` sets AlphaZero's search budget per move (default 200).
    """
    p = Path(path)
    d = p if p.is_dir() else p.parent
    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    trained_n = cfg.get("board_size")
    label = f"{cfg.get('run', d.name)} ({d.name})"

    if (d / "table.pkl.gz").exists() and (p.is_dir() or p.name == "table.pkl.gz"):
        from othello.rl.tabular import TabularLearner
        if trained_n and trained_n != env.n:
            raise ValueError(f"this table was trained on {trained_n}x{trained_n}; use --size {trained_n}")
        learner = TabularLearner.load(env, d / "table.pkl.gz")
        return LearnerAgent(learner, f"{learner.name} table {label}")

    if (d / "weights.json").exists() and (p.is_dir() or p.name == "weights.json"):
        from othello.rl.linear import LinearQLearner
        w = json.loads((d / "weights.json").read_text())
        learner = LinearQLearner(env, seed=0, features=list(w))
        learner.w[:] = list(w.values())
        return LearnerAgent(learner, f"linear model {label}")

    if (d / "model.pt").exists() and cfg.get("run") == "alphazero":
        if trained_n and trained_n != env.n:
            raise ValueError(f"this model was trained on {trained_n}x{trained_n}; use --size {trained_n}")
        import torch
        from othello.rl.alphazero.selfplay import AZAgent, build_net
        net = build_net(env.n, cfg)
        net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        return AZAgent(env, net, sims=sims or cfg.get("play_sims", 200), name=f"AlphaZero model ({d.name})")

    run_cfg = cfg.get("runs", {}).get(cfg.get("run"), {})
    if (d / "model.pt").exists() and run_cfg.get("learner") == "ppo":
        if trained_n and trained_n != env.n:
            raise ValueError(f"this model was trained on {trained_n}x{trained_n}; use --size {trained_n}")
        import torch
        from othello.rl.ppo import PPOLearner
        kw = run_cfg.get("kwargs", {})
        learner = PPOLearner(env, seed=0, hidden=kw.get("hidden", 256), layers=kw.get("layers", 2))
        learner.net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        learner.net.eval()
        return LearnerAgent(learner, f"PPO model {label}")

    if (d / "model.pt").exists():
        if trained_n and trained_n != env.n:
            raise ValueError(f"this DQN was trained on {trained_n}x{trained_n}; use --size {trained_n}")
        import torch
        from othello.rl.dqn import DQNLearner
        kw = cfg.get("runs", {}).get(cfg.get("run"), {}).get("kwargs", {})
        learner = DQNLearner(env, seed=0, hidden=kw.get("hidden", 256), layers=kw.get("layers", 2), buffer=1)
        learner.q.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        learner.q.eval()
        return LearnerAgent(learner, f"DQN model {label}")

    raise ValueError(f"no loadable model found at {path} (expected model.pt or weights.json)")
