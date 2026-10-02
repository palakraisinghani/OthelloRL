"""Bradley-Terry / Elo ratings from a round-robin of pairwise scores."""
from __future__ import annotations

import math


def fit_elo(results: dict[tuple[str, str], tuple[float, int]], anchor: str | None = None,
            iters: int = 2000) -> dict[str, float]:
    """results[(a, b)] = (points scored by a against b, games played).

    Fits Bradley-Terry strengths with the MM algorithm (draws count as half a
    point) and converts to Elo; `anchor` is pinned to 0. A small prior game
    against a virtual average opponent keeps undefeated agents finite.
    """
    players = sorted({p for pair in results for p in pair})
    wins = {p: 0.5 for p in players}            # prior: half a point...
    games = {p: {q: 0.0 for q in players} for p in players}
    for (a, b), (pts, n) in results.items():
        wins[a] += pts
        wins[b] += n - pts
        games[a][b] += n
        games[b][a] += n
    gamma = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            # ...from one game vs a virtual opponent of strength 1
            denom = 1.0 / (gamma[p] + 1.0)
            denom += sum(games[p][q] / (gamma[p] + gamma[q]) for q in players if q != p)
            new[p] = wins[p] / denom
        gamma = new
    ref = gamma[anchor] if anchor else math.exp(sum(math.log(g) for g in gamma.values()) / len(gamma))
    return {p: 400 * math.log10(gamma[p] / ref) for p in players}
