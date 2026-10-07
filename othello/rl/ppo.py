"""Proximal Policy Optimisation (Schulman et al. 2017) for self-play Othello.

Actor-critic with a shared MLP trunk over the board planes, a masked softmax
policy over all n*n+1 actions and a tanh value head. Experience is collected
from complete games (the learner's own moves only, so updates stay on-policy)
until `rollout_steps` decisions are buffered, then optimised with the clipped
surrogate objective for a few epochs.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn

# Weight decay drives many weights towards zero; once they become denormal floats (~1e-38),
# CPU matmuls/convolutions slow down ~10x (seen on the 8x8 AlphaZero run). Flush them to zero.
torch.set_flush_denormal(True)

from othello.rl.common import Learner

PLANES = ("mine", "theirs", "legal")


class PolicyValueMLP(nn.Module):
    def __init__(self, n, hidden=256, layers=2):
        super().__init__()
        dims = [len(PLANES) * n * n] + [hidden] * layers
        mods = [nn.Flatten()]
        for a, b in zip(dims, dims[1:]):
            mods += [nn.Linear(a, b), nn.ReLU()]
        self.trunk = nn.Sequential(*mods)
        self.pi = nn.Linear(hidden, n * n + 1)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        h = self.trunk(x)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, torch.tanh(self.v(h)).squeeze(-1)


class PPOLearner(Learner):
    name = "ppo"

    def __init__(self, env, seed, hidden=256, layers=2, lr=3e-4, gamma=1.0, lam=0.95, clip=0.2,
                 epochs=4, minibatch=512, rollout_steps=4096, vf_coef=0.5, ent_coef=0.01):
        super().__init__(env, seed)
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.net = PolicyValueMLP(env.n, hidden, layers)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch, self.rollout_steps = epochs, minibatch, rollout_steps
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.buf = {k: [] for k in ("obs", "mask", "act", "logp", "adv", "ret")}
        self.updates = 0
        self._log = []

    def _inputs(self, states):
        obs = np.stack([self.env.encode(s, PLANES) for s in states])
        mask = np.stack([self.env.legal_mask(s) for s in states])
        return torch.from_numpy(obs), torch.from_numpy(mask)

    @torch.no_grad()
    def act(self, state, explore):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        obs, mask = self._inputs([state])
        logits, _ = self.net(obs, mask)
        if explore:
            return int(torch.distributions.Categorical(logits=logits[0]).sample())
        return int(logits[0].argmax())

    def end_episode(self, traj, rewards):
        states = [s for s, _ in traj]
        obs, mask = self._inputs(states)
        acts = torch.tensor([a for _, a in traj])
        with torch.no_grad():
            logits, v = self.net(obs, mask)
            logp = torch.distributions.Categorical(logits=logits).log_prob(acts)
        v = v.numpy()
        T = len(traj)
        adv = np.zeros(T, np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = 0.0 if t == T - 1 else v[t + 1]
            delta = rewards[t] + self.gamma * nv - v[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        self.buf["obs"].append(obs)
        self.buf["mask"].append(mask)
        self.buf["act"].append(acts)
        self.buf["logp"].append(logp)
        self.buf["adv"].append(torch.from_numpy(adv))
        self.buf["ret"].append(torch.from_numpy(adv + v))
        if sum(len(a) for a in self.buf["act"]) >= self.rollout_steps:
            self._update()

    def _update(self):
        b = {k: torch.cat(v) for k, v in self.buf.items()}
        self.buf = {k: [] for k in self.buf}
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        n = len(adv)
        for _ in range(self.epochs):
            perm = torch.from_numpy(self.np_rng.permutation(n))
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                logits, v = self.net(b["obs"][idx], b["mask"][idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b["act"][idx])
                ratio = torch.exp(logp - b["logp"][idx])
                a = adv[idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                vf = ((v - b["ret"][idx]) ** 2).mean()
                # entropy over legal actions only (masked logits carry ~0 probability)
                ent = dist.entropy().mean()
                loss = pg + self.vf_coef * vf - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    kl = (b["logp"][idx] - logp).mean().item()
                    clipfrac = ((ratio - 1).abs() > self.clip).float().mean().item()
                self._log.append((pg.item(), vf.item(), ent.item(), kl, clipfrac))
        self.updates += 1

    def stats(self):
        if not self._log:
            return {"updates": self.updates}
        pg, vf, ent, kl, cf = np.mean(self._log, axis=0)
        self._log = []
        return {"updates": self.updates, "pg_loss": pg, "v_loss": vf, "entropy": ent,
                "approx_kl": kl, "clip_frac": cf}

    def snapshot(self, seed=0):
        return FrozenPolicyAgent(self.env, self.net, seed=seed)

    def save(self, path):
        torch.save(self.net.state_dict(), path)


class FrozenPolicyAgent:
    """A frozen copy of the policy, sampling its moves, used as a self-play opponent."""

    def __init__(self, env, net, seed=0, name="ppo-snapshot"):
        self.env, self.name = env, name
        self.net = copy.deepcopy(net).eval()
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = random.Random(seed)

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, state):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        obs = torch.from_numpy(self.env.encode(state, PLANES))[None]
        mask = torch.from_numpy(self.env.legal_mask(state))[None]
        logits, _ = self.net(obs, mask)
        return int(torch.multinomial(torch.softmax(logits[0], -1), 1, generator=self.gen))
