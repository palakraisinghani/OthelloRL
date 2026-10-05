"""Population diversity and non-transitivity measures (Phase 4).

* Policy disagreement: on a fixed set of positions, the fraction where two
  agents' greedy moves differ. Population diversity = mean over member pairs.
* Non-transitivity: directed 3-cycles A > B > C > A in a head-to-head score
  matrix, counting only edges whose 95% CI excludes 50%.
"""
from __future__ import annotations

import itertools
import math
import random

from othello.env import OthelloEnv


def position_set(env: OthelloEnv, agents: list, n: int = 1000, seed: int = 6161, sample_every: int = 3):
    """Positions from games between randomly paired `agents` (with 10% random moves for variety).

    Only positions with at least two legal moves are kept (forced moves carry no information).
    """
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a, b = rng.choice(agents), rng.choice(agents)
        s = env.initial_state()
        ply = 0
        while not env.is_terminal(s) and len(out) < n:
            moves = env.legal_moves(s)
            if len(moves) > 1 and ply % sample_every == 0:
                out.append(s)
            agent = a if s.player == 0 else b
            m = rng.choice(moves) if rng.random() < 0.1 else agent.act(s)
            s = env.step(s, m)
            ply += 1
    return out


def greedy_moves(agent, positions):
    return [agent.act(s) for s in positions]


def disagreement(moves_a, moves_b):
    return sum(x != y for x, y in zip(moves_a, moves_b)) / len(moves_a)


def mean_pairwise_disagreement(move_lists):
    pairs = list(itertools.combinations(range(len(move_lists)), 2))
    vals = [disagreement(move_lists[i], move_lists[j]) for i, j in pairs]
    return sum(vals) / len(vals), vals


def significant_beats(score, games):
    """True if `score` (of A vs B over `games`) is above 50% with a 95% CI excluding 50%."""
    half = 1.96 * math.sqrt(max(score * (1 - score), 1e-12) / games)
    return score - half > 0.5


def count_cycles(names, score, games):
    """score[(a, b)] = a's score vs b (both orders present or derivable); returns significant 3-cycles."""
    def s(a, b):
        return score[(a, b)] if (a, b) in score else 1 - score[(b, a)]

    def n(a, b):
        return games[(a, b)] if (a, b) in games else games[(b, a)]

    beats = {(a, b) for a in names for b in names if a != b and significant_beats(s(a, b), n(a, b))}
    cycles = set()
    for a, b, c in itertools.permutations(names, 3):
        if (a, b) in beats and (b, c) in beats and (c, a) in beats:
            cycles.add(tuple(sorted((a, b, c))))
    return sorted(cycles), len(beats)
