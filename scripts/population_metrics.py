"""Phase 4 metrics: policy diversity of populations and of independent self-play seeds.

  python scripts/population_metrics.py baseline      # self-play DQN seeds (Phase 3 p0)
  python scripts/population_metrics.py populations   # every finished co-evolution run

Writes results/coevo/diversity.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.agents.registry import make_agent  # noqa: E402
from othello.env import OthelloEnv  # noqa: E402
from othello.eval.population_metrics import greedy_moves, mean_pairwise_disagreement, position_set  # noqa: E402
from othello.models import load_model  # noqa: E402
from othello.utils.config import load_json  # noqa: E402

OUT = ROOT / "results/coevo/diversity.json"
W = load_json(ROOT / "results/baselines/heuristic_tuning.json")["best"]


def positions(env):
    """Fixed evaluation positions from games among the bots and earlier learners (seed 6161)."""
    agents = [make_agent({"type": "random"}, env, seed=1),
              make_agent({"type": "heuristic", "kwargs": {"weights": W, "epsilon": 0.05}}, env, seed=2),
              make_agent({"type": "minimax", "kwargs": {"depth": 2, "weights": W}}, env, seed=3),
              load_model(ROOT / "results/selfplay/ppo/seed0", env),
              load_model(ROOT / "results/classical/linear/seed0", env)]
    return position_set(env, agents, n=1000, seed=6161)


def diversity(dirs, env, pos):
    agents = [load_model(d, env) for d in dirs]
    for a in agents:
        if hasattr(a, "learner"):
            a.learner.epsilon = 0.0
            a.learner.rng.seed(0)
    mean, vals = mean_pairwise_disagreement([greedy_moves(a, pos) for a in agents])
    return {"members": [str(Path(d).relative_to(ROOT)) for d in dirs], "mean": mean, "pairs": vals}


def main(stage):
    env = OthelloEnv(6)
    pos = positions(env)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if stage == "baseline":
        res["selfplay_seeds"] = diversity([ROOT / f"results/league/dqn/p0/seed{k}" for k in (0, 1, 2)], env, pos)
        print("self-play seeds:", round(res["selfplay_seeds"]["mean"], 3))
    else:
        for run_dir in sorted((ROOT / "results/coevo").glob("F*/seed*")):
            members = sorted(run_dir.glob("m*"))
            if members and all((m / "model.pt").exists() for m in members):
                key = f"{run_dir.parent.name}/{run_dir.name}"
                res[key] = diversity(members, env, pos)
                print(key, round(res[key]["mean"], 3), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
