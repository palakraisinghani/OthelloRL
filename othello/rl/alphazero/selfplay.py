"""Self-play game generation and batched evaluation with MCTS."""
from __future__ import annotations

import numpy as np
import torch

from othello.agents.base import Agent
from othello.env import OthelloEnv
from othello.rl.alphazero.mcts import MCTS, Evaluator, Node
from othello.rl.alphazero.net import AZNet, FastEncoder


def build_net(n, cfg):
    return AZNet(n, channels=cfg["channels"], blocks=cfg["blocks"])


def play_selfplay_games(net, env, cfg, openings, rng: np.random.Generator):
    """Play len(openings) games concurrently, latest network vs itself.

    Returns training examples (planes uint8 (3,n,n), pi float32 (A,), z float32)
    for every position where the mover had a real choice.
    """
    ev = Evaluator(env, net)
    mcts = MCTS(env, ev, cfg["c_puct"], cfg["dirichlet_alpha"], cfg["noise_frac"], rng)
    enc = FastEncoder(env)
    states = []
    for op in openings:
        s = env.initial_state()
        for m in op:
            s = env.step(s, m)
        states.append(s)
    plies = [len(op) for op in openings]
    records = [[] for _ in openings]           # (planes, pi, player)
    live = [i for i, s in enumerate(states) if not env.is_terminal(s)]
    while live:
        roots, idx = [], []
        for i in live:
            moves = env.legal_moves(states[i])
            if len(moves) == 1:                 # forced move: no decision to learn
                states[i] = env.step(states[i], moves[0])
                plies[i] += 1
            else:
                roots.append(Node(states[i]))
                idx.append(i)
        if roots:
            mcts.search(roots, cfg["sims"], add_noise=True)
            for i, r in zip(idx, roots):
                records[i].append((enc.encode(states[i]).astype(np.uint8),
                                   mcts.policy(r, env.num_actions), states[i].player))
                temp = 1.0 if plies[i] < cfg["temp_plies"] else 0.0
                states[i] = env.step(states[i], mcts.choose(r, temp))
                plies[i] += 1
        live = [i for i in live if not env.is_terminal(states[i])]
    out = []
    for s, recs in zip(states, records):
        d = env.score(s)
        for planes, pi, player in recs:
            z = np.sign(d) * (1 if player == 0 else -1)
            out.append((planes, pi, np.float32(z)))
    return out, sum(len(r) for r in records)


def worker(args):
    """Process entry point: (state_dict, cfg, seed, openings) -> examples."""
    sd, cfg, seed, openings = args
    torch.set_num_threads(1)
    env = OthelloEnv(cfg["board_size"])
    net = build_net(env.n, cfg)
    net.load_state_dict(sd)
    net.eval()
    rng = np.random.default_rng(seed)
    ex = []
    chunk = cfg["concurrent_games"]
    for i in range(0, len(openings), chunk):
        e, _ = play_selfplay_games(net, env, cfg, openings[i:i + chunk], rng)
        ex += e
    return ex


def az_evaluate(net, env, opp: Agent, openings, sims, seed=0, c_puct=1.5) -> dict:
    """Greedy-MCTS player vs `opp`, every opening with both colours, games run concurrently."""
    rng = np.random.default_rng(seed)
    mcts = MCTS(env, Evaluator(env, net), c_puct=c_puct, rng=rng)
    games = []
    for op in openings:
        for colour in (0, 1):
            s = env.initial_state()
            for m in op:
                s = env.step(s, m)
            games.append([s, colour])
    live = list(range(len(games)))
    while live:
        roots, idx = [], []
        for i in live:
            s, colour = games[i]
            moves = env.legal_moves(s)
            if s.player != colour:
                games[i][0] = env.step(s, opp.act(s))
            elif len(moves) == 1:
                games[i][0] = env.step(s, moves[0])
            else:
                roots.append(Node(s))
                idx.append(i)
        if roots:
            mcts.search(roots, sims, add_noise=False)
            for i, r in zip(idx, roots):
                games[i][0] = env.step(games[i][0], mcts.choose(r, 0))
        live = [i for i in live if not env.is_terminal(games[i][0])]
    w = d = margin = 0
    for s, colour in games:
        diff = env.score(s) * (1 if colour == 0 else -1)
        w += diff > 0
        d += diff == 0
        margin += diff
    n = len(games)
    return {"win_rate": w / n, "score": (w + 0.5 * d) / n, "draw_rate": d / n, "margin": margin / n, "games": n}


class AZAgent(Agent):
    """Playable AlphaZero agent: MCTS with `sims` simulations, greedy move choice."""

    def __init__(self, env, net, sims=200, seed=None, name="alphazero", c_puct=1.5):
        super().__init__(env, seed)
        self.net, self.sims, self.name = net.eval(), sims, name
        self.mcts = MCTS(env, Evaluator(env, net), c_puct=c_puct,
                         rng=np.random.default_rng(seed))

    def act(self, state):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        root = self.mcts.search([Node(state)], self.sims, add_noise=False)[0]
        return self.mcts.choose(root, 0)


class PolicyOnlyAgent(Agent):
    """The raw network policy (argmax over legal moves), no search."""

    def __init__(self, env, net, seed=None, name="alphazero-policy"):
        super().__init__(env, seed)
        self.ev, self.name = Evaluator(env, net.eval()), name

    def act(self, state):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        (mv, pri), = self.ev([state])[0]
        return mv[int(np.argmax(pri))]
