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
    raise ValueError(f"unknown agent type {kind!r}")


def spec_name(spec: dict) -> str:
    if "name" in spec:
        return spec["name"]
    if "path" in spec:
        return str(spec["path"])
    kw = spec.get("kwargs", {})
    if spec["type"] == "minimax":
        return f"minimax-d{kw.get('depth', 4)}"
    return spec["type"]
