"""Build agents from plain dict specs so they can be shipped to worker processes."""
from __future__ import annotations

from othello.agents.heuristic import HeuristicAgent, HeuristicWeights
from othello.agents.minimax import MinimaxAgent
from othello.agents.random_agent import RandomAgent


def make_agent(spec: dict, env, seed=None):
    if "path" in spec:                      # a saved learned model (see othello/models.py)
        from othello.models import load_model
        agent = load_model(spec["path"], env)
        if hasattr(agent, "learner"):
            agent.learner.epsilon = 0.0
            agent.learner.rng.seed(seed)
        return agent
    kind = spec["type"]
    kw = dict(spec.get("kwargs", {}))
    if "weights" in kw and isinstance(kw["weights"], dict):
        kw["weights"] = HeuristicWeights(**kw["weights"])
    if kind == "random":
        return RandomAgent(env, seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(env, seed=seed, **kw)
    if kind == "minimax":
        return MinimaxAgent(env, seed=seed, **kw)
    if kind == "alphazero":                 # {"path": run dir, "sims", "temperature", "noise"}
        import json
        from pathlib import Path
        import torch
        from othello.rl.alphazero.selfplay import AZAgent, build_net
        torch.set_num_threads(1)
        d = Path(kw.pop("path"))
        cfg = json.loads((d / "config.json").read_text())
        net = build_net(env.n, cfg)
        net.load_state_dict(torch.load(d / kw.pop("model", "model.pt"), map_location="cpu"))
        return AZAgent(env, net, seed=seed, c_puct=cfg["c_puct"], dirichlet_alpha=cfg["dirichlet_alpha"], **kw)
    if kind == "edax":
        from othello.agents.edax import EdaxAgent
        return EdaxAgent(env, seed=seed, **kw)
    raise ValueError(f"unknown agent type {kind!r}")


def spec_name(spec: dict) -> str:
    if "name" in spec:
        return spec["name"]
    if "path" in spec:
        return str(spec["path"])
    kw = spec.get("kwargs", {})
    if spec["type"] == "minimax":
        return f"minimax-d{kw.get('depth', 4)}"
    if spec["type"] == "alphazero":
        return f"alphazero-s{kw.get('sims', 200)}-T{kw.get('temperature', 0)}-e{kw.get('noise', 0)}"
    if spec["type"] == "edax":
        return f"edax-l{kw.get('level', 5)}"
    return spec["type"]
