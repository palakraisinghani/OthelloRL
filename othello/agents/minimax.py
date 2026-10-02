"""Fixed-depth alpha-beta (negamax) search using the heuristic evaluation."""
from __future__ import annotations

from othello.agents.base import Agent
from othello.agents.heuristic import Evaluator, HeuristicWeights


class MinimaxAgent(Agent):
    name = "minimax"

    def __init__(self, env, depth: int = 4, weights: HeuristicWeights | None = None, seed=None):
        super().__init__(env, seed)
        self.depth = depth
        self.evaluator = Evaluator(env, weights or HeuristicWeights())
        self.nodes = 0
        # Static move ordering: corners first, then edges, interior, and
        # squares next to corners last (cheap and improves pruning a lot).
        n = env.n
        prio = {}
        for sq in range(n * n):
            r, c = divmod(sq, n)
            er, ec = min(r, n - 1 - r), min(c, n - 1 - c)
            if er == 0 and ec == 0:
                p = 0
            elif min(er, ec) == 0 and max(er, ec) >= 2:
                p = 1
            elif er >= 2 and ec >= 2:
                p = 2
            else:
                p = 3  # X- and C-squares
            prio[sq] = p
        prio[env.PASS] = 0
        self._prio = prio

    def _ordered(self, moves):
        return sorted(moves, key=self._prio.__getitem__)

    def _negamax(self, state, depth, alpha, beta):
        self.nodes += 1
        env = self.env
        if depth == 0 or env.is_terminal(state):
            return self.evaluator.evaluate(state, state.player)
        best = -float("inf")
        for m in self._ordered(env.legal_moves(state)):
            v = -self._negamax(env.step(state, m), depth - 1, -beta, -alpha)
            if v > best:
                best = v
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        return best

    def act(self, state):
        env = self.env
        moves = env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        # Shuffle then stable-sort: random order within each priority class.
        self.rng.shuffle(moves)
        moves.sort(key=self._prio.__getitem__)
        best, best_moves, alpha = -float("inf"), [], -float("inf")
        for m in moves:
            # Search with window (alpha - 1e-9) so moves tying the best value are kept exact.
            v = -self._negamax(env.step(state, m), self.depth - 1, -float("inf"), -alpha + 1e-9)
            if v > best:
                best, best_moves = v, [m]
            elif v == best:
                best_moves.append(m)
            alpha = max(alpha, v)
        return self.rng.choice(best_moves)
