"""Vanilla DQN (Mnih et al. 2015): replay buffer + periodically synced target net.

Input: raw board planes from the side to move (mine, theirs, legal moves).
Output: one Q-value per action (n*n squares + pass); illegal actions are
masked out both when acting and in the bootstrap max.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from othello.rl.common import Learner

PLANES = ("mine", "theirs", "legal")


class QNet(nn.Module):
    def __init__(self, n: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        dims = [len(PLANES) * n * n] + [hidden] * layers
        mods = []
        for a, b in zip(dims, dims[1:]):
            mods += [nn.Linear(a, b), nn.ReLU()]
        mods.append(nn.Linear(dims[-1], n * n + 1))
        self.net = nn.Sequential(nn.Flatten(), *mods)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap, obs_shape, n_actions):
        self.cap, self.i, self.full = cap, 0, False
        self.s = np.zeros((cap, *obs_shape), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, *obs_shape), np.float32)
        self.m2 = np.zeros((cap, n_actions), bool)
        self.d = np.zeros(cap, np.float32)

    def add(self, s, a, r, s2, m2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.m2[i], self.d[i] = s, a, r, s2, m2, d
        self.i = (i + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, rng, k):
        idx = rng.integers(0, len(self), k)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.m2[idx], self.d[idx]


class DQNLearner(Learner):
    name = "dqn"

    def __init__(self, env, seed, hidden=256, layers=2, lr=5e-4, gamma=1.0, buffer=200_000,
                 batch=128, train_every=4, target_sync=2_000, warmup=5_000, double=False):
        super().__init__(env, seed)
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        n = env.n
        self.q = QNet(n, hidden, layers)
        self.tq = QNet(n, hidden, layers)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.batch, self.train_every = gamma, batch, train_every
        self.target_sync, self.warmup, self.double = target_sync, warmup, double
        self.buf = ReplayBuffer(buffer, (len(PLANES), n, n), env.num_actions)
        self.steps = self.updates = 0
        self._losses = []

    def obs(self, state):
        return self.env.encode(state, PLANES)

    @torch.no_grad()
    def act(self, state, explore):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        qv = self.q(torch.from_numpy(self.obs(state))[None])[0].numpy()
        return max(moves, key=lambda m: qv[m])

    def end_episode(self, traj, rewards):
        T = len(traj)
        obs = [self.obs(s) for s, _ in traj]
        for t, (s, a) in enumerate(traj):
            done = t == T - 1
            if done:
                s2, m2 = obs[t], np.zeros(self.env.num_actions, bool)
            else:
                s2, m2 = obs[t + 1], self.env.legal_mask(traj[t + 1][0])
            self.buf.add(obs[t], a, rewards[t], s2, m2, float(done))
            self.steps += 1
            if len(self.buf) >= self.warmup and self.steps % self.train_every == 0:
                self._update()

    def _update(self):
        s, a, r, s2, m2, d = (torch.from_numpy(x) for x in self.buf.sample(self.np_rng, self.batch))
        with torch.no_grad():
            tq = self.tq(s2).masked_fill(~m2, -1e9)
            if self.double:
                a2 = self.q(s2).masked_fill(~m2, -1e9).argmax(1, keepdim=True)
                nxt = tq.gather(1, a2).squeeze(1)
            else:
                nxt = tq.max(1).values
            nxt = torch.where(d > 0, torch.zeros_like(nxt), nxt)
            target = r + self.gamma * nxt
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        self._losses.append(loss.item())
        if self.updates % self.target_sync == 0:
            self.tq.load_state_dict(self.q.state_dict())

    def stats(self):
        out = {"updates": self.updates, "loss": float(np.mean(self._losses)) if self._losses else float("nan")}
        self._losses = []
        return out

    def snapshot(self, seed=0):
        return FrozenQAgent(self.env, self.q, PLANES, seed=seed)

    def save(self, path):
        torch.save(self.q.state_dict(), path)


class FrozenQAgent:
    """A frozen copy of a Q-network used as a self-play opponent (epsilon-greedy)."""

    def __init__(self, env, net, planes, epsilon=0.05, seed=0, name="dqn-snapshot"):
        import copy
        import random
        self.env, self.planes, self.epsilon, self.name = env, planes, epsilon, name
        self.net = copy.deepcopy(net).eval()
        self.rng = random.Random(seed)

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, state):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        qv = self.net(torch.from_numpy(self.env.encode(state, self.planes))[None])[0].numpy()
        return max(moves, key=lambda m: qv[m])

