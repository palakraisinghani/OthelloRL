"""Hand-crafted evaluation and the greedy (1-ply) heuristic bot.

Features are computed from the perspective of a given player and are all
size-agnostic: square classes (corners, X-squares) are derived from geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from othello.agents.base import Agent
from othello.env import OthelloEnv, State

WIN_SCORE = 1e6


@dataclass
class HeuristicWeights:
    disc: float = 1.0       # own discs - opponent discs
    mobility: float = 5.0   # own legal moves - opponent legal moves
    corner: float = 25.0    # own corners - opponent corners
    xsquare: float = 0.0    # own - opponent discs diagonally next to an EMPTY corner

    def as_dict(self):
        return asdict(self)


class Evaluator:
    def __init__(self, env: OthelloEnv, weights: HeuristicWeights):
        self.env, self.w = env, weights
        n = env.n
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        xs = [(1, 1), (1, n - 2), (n - 2, 1), (n - 2, n - 2)]
        bit = lambda rc: 1 << (rc[0] * n + rc[1])
        self.corner_mask = sum(bit(c) for c in corners)
        self.corner_x = [(bit(c), bit(x)) for c, x in zip(corners, xs)]

    def features(self, state: State, player: int) -> dict:
        env = self.env
        mine = state.black if player == 0 else state.white
        theirs = state.white if player == 0 else state.black
        me_to_move = State(state.black, state.white, player)
        opp_to_move = State(state.black, state.white, 1 - player)
        occupied = mine | theirs
        xsq = 0
        for cb, xb in self.corner_x:
            if not occupied & cb:
                xsq += bool(mine & xb) - bool(theirs & xb)
        return {
            "disc": mine.bit_count() - theirs.bit_count(),
            "mobility": env.legal_bits(me_to_move).bit_count() - env.legal_bits(opp_to_move).bit_count(),
            "corner": (mine & self.corner_mask).bit_count() - (theirs & self.corner_mask).bit_count(),
            "xsquare": xsq,
        }

    def evaluate(self, state: State, player: int) -> float:
        """Value of `state` for `player`; exact +-WIN_SCORE (+ margin) on terminal states."""
        env = self.env
        if env.is_terminal(state):
            d = env.score(state) * (1 if player == 0 else -1)
            return (WIN_SCORE + d) if d > 0 else (-WIN_SCORE + d) if d < 0 else 0.0
        f = self.features(state, player)
        w = self.w
        return w.disc * f["disc"] + w.mobility * f["mobility"] + w.corner * f["corner"] + w.xsquare * f["xsquare"]


class HeuristicAgent(Agent):
    """Greedy one-ply bot: plays the move whose resulting position evaluates best.

    Ties are broken uniformly at random and with probability `epsilon` a random
    legal move is played, so the bot is not a single deterministic line that a
    tabular learner could memorise.
    """
    name = "heuristic"

    def __init__(self, env, weights: HeuristicWeights | None = None, epsilon: float = 0.0, seed=None):
        super().__init__(env, seed)
        self.evaluator = Evaluator(env, weights or HeuristicWeights())
        self.epsilon = epsilon

    def act(self, state):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        best, best_moves = None, []
        for m in moves:
            v = self.evaluator.evaluate(self.env.step(state, m), state.player)
            if best is None or v > best:
                best, best_moves = v, [m]
            elif v == best:
                best_moves.append(m)
        return self.rng.choice(best_moves)
