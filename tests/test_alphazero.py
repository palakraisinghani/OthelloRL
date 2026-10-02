import random

import numpy as np
import torch
import torch.nn as nn

from othello.agents import RandomAgent
from othello.env import OthelloEnv, State
from othello.eval.arena import make_openings
from othello.rl.alphazero.mcts import MCTS, Evaluator, Node
from othello.rl.alphazero.net import AZNet, FastEncoder
from othello.rl.alphazero.selfplay import az_evaluate, play_selfplay_games
from othello.rl.alphazero.train import Augmenter
from othello.symmetry import get_symmetry


class UniformNet(nn.Module):
    """Uniform priors, zero value: MCTS then reduces to pure tree search on terminal results."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        B = x.shape[0]
        return torch.zeros(B, self.n * self.n + 1), torch.zeros(B)


def test_fast_encoder_matches_env_encode():
    env = OthelloEnv(6)
    enc = FastEncoder(env)
    rng = random.Random(0)
    s = env.initial_state()
    for _ in range(10):
        assert np.array_equal(enc.encode(s), env.encode(s, ("mine", "theirs", "legal")))
        s = env.step(s, rng.choice(env.legal_moves(s)))


def test_net_shapes():
    net = AZNet(6, channels=16, blocks=2)
    p, v = net(torch.zeros(5, 3, 6, 6))
    assert p.shape == (5, 37) and v.shape == (5,)
    assert (v.abs() <= 1).all()


def test_augmenter_matches_transformed_positions():
    env, sym = OthelloEnv(6), get_symmetry(6)
    enc = FastEncoder(env)
    rng = random.Random(1)
    s = env.initial_state()
    for _ in range(5):
        s = env.step(s, rng.choice(env.legal_moves(s)))
    x = enc.encode(s)[None]
    pi = np.zeros((1, 37), np.float32)
    move = env.legal_moves(s)[0]
    pi[0, move] = 1
    aug = Augmenter(6, np.random.default_rng(0))
    for _ in range(20):
        x2, pi2 = aug(x.copy(), pi.copy())
        # find which transform was applied by matching against all 8
        matches = []
        for t in range(8):
            st = State(sym.transform_bits(s.black, t), sym.transform_bits(s.white, t), s.player)
            if np.array_equal(enc.encode(st), x2[0]):
                matches.append(t)
        assert matches
        assert any(pi2[0, sym.transform_action(move, t)] == 1 for t in matches)


def test_mcts_takes_immediate_win():
    env = OthelloEnv(4)
    bit = lambda r, c: 1 << (r * 4 + c)
    # black a1, white b1: c1 captures white's only disc and wins
    s = State(black=bit(0, 0), white=bit(0, 1), player=0)
    mcts = MCTS(env, Evaluator(env, UniformNet(4)), rng=np.random.default_rng(0))
    root = mcts.search([Node(s)], 50, add_noise=False)[0]
    assert mcts.choose(root, 0) == 2


def test_backup_signs():
    """A child that is a win for the side to move there must look bad from the parent."""
    env = OthelloEnv(4)
    bit = lambda r, c: 1 << (r * 4 + c)
    s = State(black=bit(0, 0), white=bit(0, 1), player=0)
    mcts = MCTS(env, Evaluator(env, UniformNet(4)), rng=np.random.default_rng(0))
    root = mcts.search([Node(s)], 20, add_noise=False)[0]
    i = root.moves.index(2)
    # the win is only final after both sides pass (two plies deeper), so the edge
    # converges toward +1 from black's perspective and dominates the visits
    assert root.W[i] / root.N[i] > 0.8
    assert root.N[i] == max(root.N)


class CornerMobilityNet(UniformNet):
    """Uniform priors; value = squashed corner difference + own mobility (side to move)."""

    def forward(self, x):
        p, _ = super().forward(x)
        m, t = x[:, 0], x[:, 1]
        corners = lambda b: b[:, 0, 0] + b[:, 0, -1] + b[:, -1, 0] + b[:, -1, -1]
        return p, torch.tanh((corners(m) - corners(t)) / 2 + x[:, 2].sum((1, 2)) / 20)


def test_mcts_with_simple_value_beats_random():
    env = OthelloEnv(6)
    ops = make_openings(env, 10, 4, 0)
    r = az_evaluate(CornerMobilityNet(6), env, RandomAgent(env, seed=0), ops, sims=48, seed=0)
    assert r["score"] > 0.75


def test_selfplay_examples_are_well_formed():
    env = OthelloEnv(6)
    cfg = {"c_puct": 1.5, "dirichlet_alpha": 0.5, "noise_frac": 0.25, "sims": 8, "temp_plies": 4}
    ex, n = play_selfplay_games(UniformNet(6), env, cfg, [[], []], np.random.default_rng(0))
    assert n == len(ex) > 10
    for planes, pi, z in ex:
        assert planes.shape == (3, 6, 6) and abs(pi.sum() - 1) < 1e-5 and z in (-1, 0, 1)
        assert (pi[:36][planes[2].reshape(-1) == 0] == 0).all()   # mass only on legal moves
