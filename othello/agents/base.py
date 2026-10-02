"""Common agent interface: every player (bot, RL policy, MCTS) implements act()."""
from __future__ import annotations

import random

from othello.env import OthelloEnv, State


class Agent:
    name = "agent"

    def __init__(self, env: OthelloEnv, seed: int | None = None):
        self.env = env
        self.rng = random.Random(seed)

    def act(self, state: State) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of each game (for agents with per-game state)."""
