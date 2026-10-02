"""Generate TikZ board diagrams that illustrate the rules (report Section 2).

Every position is produced by the real engine (othello.env), so each diagram is
guaranteed to be a legal, correctly-flipped Othello position.

  python scripts/make_rule_diagrams.py      # writes report/figures/rules_*.tex
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from othello.env import BLACK, WHITE, OthelloEnv, State  # noqa: E402

OUT = ROOT / "report" / "figures"
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
COLS = "abcdefghij"

STYLE = r"""\tikzset{
  board/.style={fill=felt, draw=feltdark, line width=1.2pt},
  grid/.style={feltdark, line width=0.5pt},
  bdisc/.style={circle, inner sep=0pt, minimum size=7.4mm, ball color=black!85, draw=black},
  wdisc/.style={circle, inner sep=0pt, minimum size=7.4mm, ball color=white, draw=black!55},
  shadow/.style={circle, inner sep=0pt, minimum size=7.4mm, fill=black!45, opacity=0.35},
  ghostb/.style={circle, inner sep=0pt, minimum size=6mm, draw=black, dashed, line width=0.9pt},
  ghostw/.style={circle, inner sep=0pt, minimum size=6mm, draw=white, dashed, line width=0.9pt},
  newmark/.style={circle, inner sep=0pt, minimum size=8.6mm, draw=red!75!black, line width=1.6pt},
  flipmark/.style={circle, inner sep=0pt, minimum size=8.6mm, draw=orange!95!black, line width=1.4pt},
  capture/.style={-{Stealth[length=2.2mm]}, red!75!black, line width=1.6pt, opacity=0.85, shorten >=1.5mm, shorten <=3.2mm},
  illegal/.style={red!75!black, line width=2pt},
  coord/.style={font=\large\sffamily, text=black!75},
  halo/.style={white, line width=3.4pt, opacity=0.75, shorten >=1.5mm, shorten <=3.2mm},
}"""


def sq_xy(n, sq):
    r, c = divmod(sq, n)
    return c + 0.5, n - r - 0.5          # row 1 at the top


def name(n, sq):
    r, c = divmod(sq, n)
    return f"{COLS[c]}{r + 1}"


def capture_lines(env, state, move):
    """[(flipped squares in order, anchor square)] for each direction the move captures in."""
    n = env.n
    own, opp = state.own(), state.opp()
    r0, c0 = divmod(move, n)
    lines = []
    for dr, dc in DIRS:
        r, c, run = r0 + dr, c0 + dc, []
        while 0 <= r < n and 0 <= c < n and opp >> (r * n + c) & 1:
            run.append(r * n + c)
            r, c = r + dr, c + dc
        if run and 0 <= r < n and 0 <= c < n and own >> (r * n + c) & 1:
            lines.append((run, r * n + c))
    return lines


def tikz(env, state, ghosts=(), new=None, flipped=(), arrows=(), illegal=(), scale=0.62, label_mover=False):
    n = env.n
    L = [r"\begin{tikzpicture}[scale=%s, every node/.style={transform shape}]" % scale]
    L.append(rf"\fill[board] (0,0) rectangle ({n},{n});")
    for i in range(1, n):
        L.append(rf"\draw[grid] ({i},0) -- ({i},{n}); \draw[grid] (0,{i}) -- ({n},{i});")
    L.append(rf"\draw[feltdark, line width=1.2pt] (0,0) rectangle ({n},{n});")
    for c in range(n):
        L.append(rf"\node[coord] at ({c + 0.5},{n + 0.4}) {{{COLS[c]}}};")
    for r in range(n):
        L.append(rf"\node[coord] at (-0.4,{n - r - 0.5}) {{{r + 1}}};")
    for sq in range(n * n):
        x, y = sq_xy(n, sq)
        if state.black >> sq & 1 or state.white >> sq & 1:
            L.append(rf"\node[shadow] at ({x + 0.06},{y - 0.07}) {{}};")
            L.append(rf"\node[{'bdisc' if state.black >> sq & 1 else 'wdisc'}] at ({x},{y}) {{}};")
    ghost_style = "ghostb" if state.player == BLACK else "ghostw"
    for sq in ghosts:
        x, y = sq_xy(n, sq)
        L.append(rf"\node[{ghost_style}] at ({x},{y}) {{}};")
    for sq in flipped:
        x, y = sq_xy(n, sq)
        L.append(rf"\node[flipmark] at ({x},{y}) {{}};")
    if new is not None:
        x, y = sq_xy(n, new)
        L.append(rf"\node[newmark] at ({x},{y}) {{}};")
    for a, b in arrows:
        (x0, y0), (x1, y1) = sq_xy(n, a), sq_xy(n, b)
        L.append(rf"\draw[halo] ({x0},{y0}) -- ({x1},{y1});")
        L.append(rf"\draw[capture] ({x0},{y0}) -- ({x1},{y1});")
    for sq in illegal:
        x, y = sq_xy(n, sq)
        L.append(rf"\draw[illegal] ({x - 0.28},{y - 0.28}) -- ({x + 0.28},{y + 0.28}) "
                 rf"({x - 0.28},{y + 0.28}) -- ({x + 0.28},{y - 0.28});")
    L.append(r"\end{tikzpicture}")
    return "\n".join(L)


def find_multi_capture(env, seed=3, min_dirs=3, max_discs=18):
    rng = random.Random(seed)
    for _ in range(5000):
        s = env.initial_state()
        while not env.is_terminal(s):
            if s.player == BLACK and (s.black | s.white).bit_count() <= max_discs:
                for m in env.legal_moves(s):
                    if m != env.PASS and len(capture_lines(env, s, m)) >= min_dirs:
                        return s, m
            s = env.step(s, rng.choice(env.legal_moves(s)))
    raise RuntimeError("no multi-capture position found")


def find_pass(env, seed=5):
    """Earliest-found position where the side to move must pass but the opponent can move."""
    rng = random.Random(seed)
    best = None
    for _ in range(4000):
        s = env.initial_state()
        while not env.is_terminal(s):
            moves = env.legal_moves(s)
            if moves == [env.PASS]:
                other = State(s.black, s.white, 1 - s.player)
                discs = (s.black | s.white).bit_count()
                if env.legal_moves(other) != [env.PASS] and (best is None or discs < best[1]):
                    best = (s, discs)
                break
            s = env.step(s, rng.choice(moves))
    return best[0]


def find_end(env, seed=11):
    """A finished game with a full board and a close score."""
    rng = random.Random(seed)
    for _ in range(4000):
        s = env.initial_state()
        while not env.is_terminal(s):
            s = env.step(s, rng.choice(env.legal_moves(s)))
        b, w = env.counts(s)
        if b + w == env.n * env.n and abs(b - w) == 4:
            return s
    raise RuntimeError


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = OthelloEnv(6)
    n = env.n
    sq = lambda nm: (int(nm[1:]) - 1) * n + COLS.index(nm[0])
    facts = {}

    # (a) start position, Black's legal moves, one illegal square
    s0 = env.initial_state()
    legal = env.legal_moves(s0)
    bad = sq("e5")
    assert bad not in legal
    a = tikz(env, s0, ghosts=legal, illegal=[bad])

    # (b) simple capture: Black c2 flips c3
    m = sq("c2")
    lines = capture_lines(env, s0, m)
    s1 = env.step(s0, m)
    flips = [q for run, _ in lines for q in run]
    b = tikz(env, s1, new=m, flipped=flips, arrows=[(m, anchor) for _, anchor in lines])
    facts["simple"] = (name(n, m), [name(n, q) for q in flips], [name(n, an) for _, an in lines])

    # (c, d) multi-direction capture, before and after
    s2, m2 = find_multi_capture(env)
    lines2 = capture_lines(env, s2, m2)
    after = env.step(s2, m2)
    flips2 = [q for run, _ in lines2 for q in run]
    c = tikz(env, s2, ghosts=[m2], arrows=[(m2, anchor) for _, anchor in lines2])
    d = tikz(env, after, new=m2, flipped=flips2)
    facts["multi"] = (name(n, m2), len(lines2), [name(n, q) for q in flips2])

    # (e) forced pass
    sp = find_pass(env)
    mover = "Black" if sp.player == BLACK else "White"
    other_moves = env.legal_moves(State(sp.black, sp.white, 1 - sp.player))
    e = tikz(env, State(sp.black, sp.white, 1 - sp.player), ghosts=other_moves)
    facts["pass"] = (mover, "White" if mover == "Black" else "Black", len(other_moves))

    # (f) final position
    se = find_end(env)
    f = tikz(env, se)
    facts["end"] = env.counts(se)

    for key, body in zip("abcdef", [a, b, c, d, e, f]):
        (OUT / f"rules_{key}.tex").write_text(body + "\n")
    (OUT / "rules_style.tex").write_text(STYLE + "\n")
    m_name, k, fl = facts["multi"]
    mv, oth, nopp = facts["pass"]
    bb, ww = facts["end"]
    macros = [
        rf"\newcommand{{\RuleMultiMove}}{{{m_name}}}",
        rf"\newcommand{{\RuleMultiDirs}}{{{k}}}",
        rf"\newcommand{{\RuleMultiFlips}}{{{len(fl)}}}",
        rf"\newcommand{{\RulePassMover}}{{{mv}}}",
        rf"\newcommand{{\RulePassOther}}{{{oth}}}",
        rf"\newcommand{{\RuleEndBlack}}{{{bb}}}",
        rf"\newcommand{{\RuleEndWhite}}{{{ww}}}",
    ]
    (OUT / "rules_meta.tex").write_text("\n".join(macros) + "\n")
    print(facts)


if __name__ == "__main__":
    main()
