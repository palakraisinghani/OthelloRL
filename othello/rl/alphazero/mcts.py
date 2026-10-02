"""PUCT Monte Carlo tree search, batched across many independent games.

Each simulation descends every tree once, collects the leaves that need a
network evaluation, evaluates them in a single forward pass, then expands and
backs up. Values are always from the perspective of the player to move at the
node; since Othello alternates the side to move on every action (passes
included), a value flips sign at each step up the tree.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from othello.rl.alphazero.net import FastEncoder


class Node:
    __slots__ = ("state", "moves", "P", "N", "W", "children", "expanded", "terminal", "value", "sumN")

    def __init__(self, state):
        self.state = state
        self.moves = None
        self.P = self.N = self.W = None
        self.children = None
        self.expanded = False
        self.terminal = False
        self.value = 0.0
        self.sumN = 0


def terminal_value(env, state):
    """Exact result for the side to move at a terminal state."""
    d = env.score(state) * (1 if state.player == 0 else -1)
    return 1.0 if d > 0 else -1.0 if d < 0 else 0.0


class Evaluator:
    """Batched network evaluation: priors over legal moves and a value per state."""

    def __init__(self, env, net):
        self.env, self.net = env, net
        self.enc = FastEncoder(env)
        self.calls = 0

    @torch.no_grad()
    def __call__(self, states):
        self.calls += 1
        x = torch.from_numpy(np.stack([self.enc.encode(s) for s in states]))
        logits, v = self.net(x)
        logits = logits.numpy()
        out = []
        for i, s in enumerate(states):
            moves = self.env.legal_moves(s)
            lg = logits[i, moves]
            p = np.exp(lg - lg.max())
            out.append((moves, (p / p.sum()).tolist()))
        return out, v.numpy().tolist()


class MCTS:
    def __init__(self, env, evaluator, c_puct=1.5, dirichlet_alpha=0.5, noise_frac=0.25, rng=None):
        self.env, self.ev = env, evaluator
        self.c_puct, self.alpha, self.eps = c_puct, dirichlet_alpha, noise_frac
        self.rng = rng or np.random.default_rng()

    def _expand(self, node, moves, priors):
        node.moves = moves
        node.P = priors
        node.N = [0] * len(moves)
        node.W = [0.0] * len(moves)
        node.children = [None] * len(moves)
        node.expanded = True

    def _select(self, node):
        sq = self.c_puct * math.sqrt(node.sumN + 1)
        best, best_i = -1e9, 0
        N, W, P = node.N, node.W, node.P
        for i in range(len(P)):
            n = N[i]
            u = (W[i] / n if n else 0.0) + sq * P[i] / (1 + n)
            if u > best:
                best, best_i = u, i
        return best_i

    def search(self, roots, sims, add_noise):
        """Run `sims` simulations from each root (Node). Returns the roots."""
        env = self.env
        fresh = [r for r in roots if not r.expanded and not env.is_terminal(r.state)]
        if fresh:
            outs, _ = self.ev([r.state for r in fresh])
            for r, (moves, pri) in zip(fresh, outs):
                self._expand(r, moves, pri)
        if add_noise:
            for r in roots:
                if r.expanded and len(r.moves) > 1:
                    noise = self.rng.dirichlet([self.alpha] * len(r.moves))
                    r.P = [(1 - self.eps) * p + self.eps * z for p, z in zip(r.P, noise)]
        for _ in range(sims):
            pending, paths = [], []
            for root in roots:
                if not root.expanded:
                    continue
                node, path = root, []
                while node.expanded:
                    i = self._select(node)
                    path.append((node, i))
                    child = node.children[i]
                    if child is None:
                        child = Node(env.step(node.state, node.moves[i]))
                        if env.is_terminal(child.state):
                            child.terminal = True
                            child.value = terminal_value(env, child.state)
                        node.children[i] = child
                    node = child
                    if node.terminal:
                        break
                if node.terminal:
                    self._backup(path, node.value)
                else:
                    pending.append(node)
                    paths.append(path)
            if pending:
                outs, vals = self.ev([nd.state for nd in pending])
                for nd, path, (moves, pri), v in zip(pending, paths, outs, vals):
                    self._expand(nd, moves, pri)
                    self._backup(path, v)
        return roots

    @staticmethod
    def _backup(path, v):
        for node, i in reversed(path):
            v = -v
            node.N[i] += 1
            node.W[i] += v
            node.sumN += 1

    def policy(self, root, num_actions):
        """Visit-count distribution over the full action space."""
        pi = np.zeros(num_actions, np.float32)
        total = sum(root.N)
        for m, n in zip(root.moves, root.N):
            pi[m] = n / total if total else 1.0 / len(root.moves)
        return pi

    def choose(self, root, temperature):
        N = np.array(root.N, dtype=np.float64)
        if temperature == 0 or N.sum() == 0:
            best = np.flatnonzero(N == N.max())
            return root.moves[int(self.rng.choice(best))]
        p = N ** (1.0 / temperature)
        return root.moves[int(self.rng.choice(len(p), p=p / p.sum()))]
