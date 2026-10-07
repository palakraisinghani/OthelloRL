"""Edax (https://github.com/abulmo/edax-reversi), a world-class 8x8 Othello engine, as an Agent.

Built from source into third_party/edax (see PLAN Step 5.4). Each agent drives one
Edax process through its native text interface: `setboard` with the current
position, then `go`, then parse "Edax plays XY". Strength is set by the search
`level` (depth). The opening book is off, so play comes from search alone.
Only 8x8 is supported.
"""
from __future__ import annotations

import os
import select
import subprocess
import time
from pathlib import Path

from othello.agents.base import Agent

EDAX_DIR = Path(__file__).resolve().parents[2] / "third_party" / "edax"
EDAX_BIN = EDAX_DIR / "bin" / "lEdax-x86-64-v2"


class EdaxAgent(Agent):
    name = "edax"

    def __init__(self, env, level: int = 5, seed=None, timeout: float = 60.0):
        super().__init__(env, seed)
        if env.n != 8:
            raise ValueError("Edax plays 8x8 only")
        if not EDAX_BIN.exists():
            raise FileNotFoundError(f"Edax binary not found at {EDAX_BIN} (see PLAN Step 5.4)")
        self.level, self.timeout = level, timeout
        self.name = f"edax-l{level}"
        self.proc = subprocess.Popen([str(EDAX_BIN), "-n", "1", "-book-usage", "off", "-l", str(level),
                                      "-verbose", "0"], cwd=EDAX_DIR, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._read()
        self._cmd("mode 3")          # Edax never moves on its own

    def _read(self):
        buf, end = "", time.time() + self.timeout
        fd = self.proc.stdout.fileno()
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                chunk = os.read(fd, 65536).decode()
                if not chunk:
                    raise RuntimeError("Edax exited unexpectedly")
                buf += chunk
                if buf.rstrip().endswith(">"):
                    return buf
        raise TimeoutError(f"Edax did not answer within {self.timeout}s: {buf[-200:]!r}")

    def _cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self._read()

    def board_string(self, state):
        cells = []
        for sq in range(64):
            b = 1 << sq
            cells.append("*" if state.black & b else "O" if state.white & b else "-")
        return "".join(cells) + (" *" if state.player == 0 else " O")

    def act(self, state):
        moves = self.env.legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        self._cmd("setboard " + self.board_string(state))
        out = self._cmd("go")
        for line in out.splitlines():
            if "Edax plays" in line:
                mv = line.split("Edax plays")[1].strip()[:2].lower()
                sq = (int(mv[1]) - 1) * 8 + "abcdefgh".index(mv[0])
                if sq not in moves:
                    raise RuntimeError(f"Edax played illegal move {mv} in {self.board_string(state)}")
                return sq
        raise RuntimeError(f"no move in Edax output: {out[-200:]!r}")

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
