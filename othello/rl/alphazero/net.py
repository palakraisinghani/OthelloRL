"""Residual policy-value network (AlphaZero, Silver et al. 2018), sized for small boards.

Input: board planes from the side to move. Output: policy logits over the
n*n squares + pass, and a scalar value in [-1, 1] (expected result for the side
to move). Illegal actions are masked by the caller.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Weight decay drives many weights towards zero; once they become denormal floats (~1e-38),
# CPU matmuls/convolutions slow down ~10x (seen on the 8x8 AlphaZero run). Flush them to zero.
torch.set_flush_denormal(True)

PLANES = ("mine", "theirs", "legal")


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(h)))


class AZNet(nn.Module):
    def __init__(self, n, channels=64, blocks=4, in_planes=len(PLANES)):
        super().__init__()
        self.n = n
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.p_head = nn.Sequential(nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(2 * n * n, n * n + 1))
        self.v_head = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(n * n, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.p_head(h), self.v_head(h).squeeze(-1)


class FastEncoder:
    """Bitboards -> (3, n, n) float32 planes [mine, theirs, legal] without Python bit loops."""

    def __init__(self, env):
        self.env = env
        self.n = env.n
        self.nbytes = (env.n * env.n + 7) // 8

    def _bits(self, x):
        b = np.frombuffer(x.to_bytes(self.nbytes, "little"), dtype=np.uint8)
        return np.unpackbits(b, bitorder="little")[: self.n * self.n]

    def encode(self, state):
        n = self.n
        own, opp = state.own(), state.opp()
        out = np.empty((3, n * n), np.float32)
        out[0] = self._bits(own)
        out[1] = self._bits(opp)
        out[2] = self._bits(self.env.legal_bits(state))
        return out.reshape(3, n, n)


def count_params(net):
    return sum(p.numel() for p in net.parameters())
