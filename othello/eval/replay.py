"""Save game records as JSONL and replay them as ASCII boards."""
from __future__ import annotations

import json
from pathlib import Path

from othello.env import OthelloEnv


def save_games(path, games, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for g in games:
            f.write(json.dumps({**(meta or {}), **g}) + "\n")


def load_games(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def replay(env: OthelloEnv, game: dict) -> str:
    """Render every position of a recorded game."""
    s = env.initial_state()
    frames = []
    for m in game["opening"]:
        s = env.step(s, m)
    frames.append("after opening " + " ".join(env.action_name(m) for m in game["opening"]))
    frames.append(env.render(s))
    for i, m in enumerate(game["moves"], 1):
        s = env.step(s, m)
        frames.append(f"move {i}: {env.action_name(m)}")
        frames.append(env.render(s))
    b, w = env.counts(s)
    frames.append(f"final: black {b} - white {w}")
    return "\n".join(frames)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    g = load_games(a.path)[a.index]
    print(replay(OthelloEnv(g["board_size"]), g))
