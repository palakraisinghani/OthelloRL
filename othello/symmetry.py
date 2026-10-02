"""The 8 dihedral symmetries of an n x n board, applied to bitboards and actions.

Bitboards are transformed with byte-wise lookup tables (fast in pure Python).
`canonical(own, opp)` returns the lexicographically smallest image of a
position and the transform index that produced it, so tabular learners can
share values between symmetric positions.
"""
from __future__ import annotations

from functools import lru_cache


def _square_perms(n: int) -> list[list[int]]:
    perms = []
    for t in range(8):
        p = []
        for sq in range(n * n):
            r, c = divmod(sq, n)
            if t & 4:
                r, c = c, r                 # transpose
            if t & 1:
                r = n - 1 - r               # flip vertical
            if t & 2:
                c = n - 1 - c               # flip horizontal
            p.append(r * n + c)
        perms.append(p)
    return perms


class Symmetry:
    def __init__(self, n: int):
        self.n = n
        self.nsq = n * n
        self.perms = _square_perms(n)                     # perms[t][sq] -> image square
        self.inv = [[0] * self.nsq for _ in range(8)]
        for t, p in enumerate(self.perms):
            for sq, im in enumerate(p):
                self.inv[t][im] = sq
        nbytes = (self.nsq + 7) // 8
        self.nbytes = nbytes
        self.tables = []
        for p in self.perms:
            tt = []
            for k in range(nbytes):
                row = []
                for v in range(256):
                    out = 0
                    for b in range(8):
                        sq = 8 * k + b
                        if v >> b & 1 and sq < self.nsq:
                            out |= 1 << p[sq]
                    row.append(out)
                tt.append(row)
            self.tables.append(tt)

    def transform_bits(self, bits: int, t: int) -> int:
        tt = self.tables[t]
        out = 0
        k = 0
        while bits:
            out |= tt[k][bits & 0xFF]
            bits >>= 8
            k += 1
        return out

    def transform_action(self, action: int, t: int) -> int:
        return action if action >= self.nsq else self.perms[t][action]

    def inverse_action(self, action: int, t: int) -> int:
        return action if action >= self.nsq else self.inv[t][action]

    def canonical(self, own: int, opp: int) -> tuple[tuple[int, int], int]:
        best, best_t = None, 0
        for t in range(8):
            key = (self.transform_bits(own, t), self.transform_bits(opp, t))
            if best is None or key < best:
                best, best_t = key, t
        return best, best_t


@lru_cache(maxsize=None)
def get_symmetry(n: int) -> Symmetry:
    return Symmetry(n)
