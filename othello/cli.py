"""`othello` — play Othello in the terminal against bots or trained models."""
from __future__ import annotations

import argparse
import os
import random
import sys
import textwrap
from pathlib import Path

from othello.agents.heuristic import HeuristicAgent, HeuristicWeights
from othello.agents.minimax import MinimaxAgent
from othello.agents.random_agent import RandomAgent
from othello.env import BLACK, WHITE, OthelloEnv

# Tuned on 6x6 (results/baselines/heuristic_tuning.json).
TUNED = HeuristicWeights(disc=1, mobility=5, corner=50, xsquare=-25)

LEVELS = {
    1: ("beginner", "plays uniformly random legal moves", lambda env, s: RandomAgent(env, seed=s)),
    2: ("easy", "grabs as many discs as it can (greedy)",
        lambda env, s: HeuristicAgent(env, HeuristicWeights(disc=1, mobility=0, corner=0, xsquare=0), seed=s)),
    3: ("medium", "greedy, values corners and mobility",
        lambda env, s: HeuristicAgent(env, TUNED, seed=s)),
    4: ("hard", "alpha-beta search, 3 moves ahead", lambda env, s: MinimaxAgent(env, depth=3, weights=TUNED, seed=s)),
    5: ("expert", "alpha-beta search, 5 moves ahead", lambda env, s: MinimaxAgent(env, depth=5, weights=TUNED, seed=s)),
}
LEVEL_NAMES = {name: k for k, (name, _, _) in LEVELS.items()}
COLS = "abcdefghijklmnop"

# The project's final trained model. models/oroletto.json points at its run
# directory once it exists: {"path": "...", "board_size": N, "about": "..."}.
FINAL_MODEL_NAME = "Oroletto"
FINAL_MODEL_POINTER = Path(__file__).resolve().parents[1] / "models" / "oroletto.json"
# Interim suggestion until Oroletto is trained (strongest classical-RL model so far).
INTERIM_MODEL = Path(__file__).resolve().parents[1] / "results/classical/dqn/seed0"


def load_oroletto(env):
    import json
    from othello.models import load_model
    if not FINAL_MODEL_POINTER.exists():
        raise ValueError(
            f"{FINAL_MODEL_NAME} has not been trained yet. It will be the project's final self-play model.\n"
            f"  For now, try the strongest model so far:  othello --model {INTERIM_MODEL}\n"
            f"  or the strongest bot:                    othello --play expert")
    info = json.loads(FINAL_MODEL_POINTER.read_text())
    path = Path(info["path"])
    if not path.is_absolute():
        path = FINAL_MODEL_POINTER.parents[1] / path
    if info.get("board_size") and info["board_size"] != env.n:
        raise ValueError(f"{FINAL_MODEL_NAME} plays {info['board_size']}x{info['board_size']}; "
                         f"use --size {info['board_size']}")
    agent = load_model(path, env, sims=info.get("play_sims"))
    agent.name = FINAL_MODEL_NAME
    return agent


# ---------------------------------------------------------------- rendering
class Style:
    def __init__(self, color: bool):
        self.color = color

    def _c(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def black(self):
        return self._c("1;30;42", "●") if self.color else "X"

    def white(self):
        return self._c("1;97;42", "●") if self.color else "O"

    def empty(self):
        return self._c("32;42", "·") if self.color else "."

    def legal(self):
        return self._c("1;33;42", "∗") if self.color else "*"

    def last(self, s):
        return self._c("4", s) if self.color else s

    def bold(self, s):
        return self._c("1", s)

    def dim(self, s):
        return self._c("2", s)

    def good(self, s):
        return self._c("32", s)

    def bad(self, s):
        return self._c("31", s)


def render(env, state, st: Style, show_legal=True, last=None, highlight=()):
    n = env.n
    legal = env.legal_bits(state) if show_legal and not env.is_terminal(state) else 0
    lines = ["    " + " ".join(COLS[:n])]
    for r in range(n):
        cells = []
        for c in range(n):
            sq = r * n + c
            b = 1 << sq
            if state.black & b:
                cell = st.black()
            elif state.white & b:
                cell = st.white()
            elif legal & b:
                cell = st.legal()
            else:
                cell = st.empty()
            if sq == last or sq in highlight:
                cell = st.last(cell) if st.color else cell.lower()   # plain: x/o = just changed
            sep = st._c("42", " ") if st.color and c < n - 1 else " "
            cells.append(cell + (sep if c < n - 1 else ""))
        lines.append(f"{r + 1:>3} " + "".join(cells) + f" {r + 1}")
    lines.append("    " + " ".join(COLS[:n]))
    b, w = env.counts(state)
    lines.append(f"    {st.black()} Black {b:>2}   {st.white()} White {w:>2}")
    return "\n".join(lines)


def parse_move(env, text):
    t = text.strip().lower()
    if t in ("pass", "p"):
        return env.PASS
    if len(t) >= 2 and t[0] in COLS[: env.n] and t[1:].isdigit():
        r, c = int(t[1:]) - 1, COLS.index(t[0])
        if 0 <= r < env.n:
            return r * env.n + c
    if len(t) >= 2 and t[-1] in COLS[: env.n] and t[:-1].isdigit():   # also accept "4c"
        r, c = int(t[:-1]) - 1, COLS.index(t[-1])
        if 0 <= r < env.n:
            return r * env.n + c
    return None


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        print()
        raise SystemExit(0)
    except KeyboardInterrupt:
        print("\nbye!")
        raise SystemExit(0)


GAME_HELP = """\
  Commands:
    c4        place a disc (column letter + row number, e.g. c4 or d3)
    hint      ask the expert bot for a suggestion
    moves     list your legal moves
    undo      take back your last move (and the bot's reply)
    board     redraw the board
    help      show this list
    quit      leave the game
  Squares marked {legal} are your legal moves."""


# ------------------------------------------------------------------- game
def play_game(env, opponent, opp_name, human_colour, st: Style, hints=True, coach=False, rng=None):
    rng = rng or random.Random()
    s = env.initial_state()
    history = [(s, None)]
    last = None
    hinter = MinimaxAgent(env, depth=4, weights=TUNED, seed=rng.randrange(10**9))
    you = "Black ●" if human_colour == BLACK else "White ○"
    print(f"\nYou are {st.bold(you if st.color else ('Black (X)' if human_colour == BLACK else 'White (O)'))} "
          f"vs {st.bold(opp_name)}. Black moves first. Type 'help' for commands.\n")
    print(render(env, s, st, hints, last))
    while not env.is_terminal(s):
        moves = env.legal_moves(s)
        if s.player != human_colour:
            a = opponent.act(s)
            s = env.step(s, a)
            last = None if a == env.PASS else a
            history.append((s, a))
            print(f"\n{opp_name} plays {st.bold(env.action_name(a))}"
                  + (" (no legal move, must pass)" if a == env.PASS else ""))
            print(render(env, s, st, hints, last))
            continue
        if moves == [env.PASS]:
            print(st.dim("\nYou have no legal move and must pass."))
            ask("  press Enter to pass ")
            s = env.step(s, env.PASS)
            history.append((s, env.PASS))
            continue
        cmd = ask(f"\nyour move> ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            print("Game abandoned.")
            return None
        if cmd in ("h", "help", "?"):
            print(GAME_HELP.format(legal=st.legal()))
            continue
        if cmd in ("b", "board"):
            print(render(env, s, st, hints, last))
            continue
        if cmd in ("m", "moves"):
            print("  legal moves: " + ", ".join(env.action_name(m) for m in moves))
            continue
        if cmd == "hint":
            h = hinter.act(s)
            print(f"  hint: try {st.bold(env.action_name(h))}")
            continue
        if cmd in ("u", "undo"):
            # pop back to the previous position where it was the human's turn
            while len(history) > 1:
                history.pop()
                if history[-1][0].player == human_colour and len(env.legal_moves(history[-1][0])) and \
                        env.legal_moves(history[-1][0]) != [env.PASS]:
                    break
            s = history[-1][0]
            last = history[-1][1] if history[-1][1] not in (None, env.PASS) else None
            print(render(env, s, st, hints, last))
            continue
        a = parse_move(env, cmd)
        if a is None or a not in moves:
            print(st.bad(f"  '{cmd}' is not a legal move.") + " Legal: "
                  + ", ".join(env.action_name(m) for m in moves) + "  (type 'help' for commands)")
            continue
        if coach:
            best = hinter.act(s)
            if best != a:
                print(st.dim(f"  coach: the expert bot would have played {env.action_name(best)}"))
        s = env.step(s, a)
        last = a
        history.append((s, a))
        print(render(env, s, st, hints, last))

    b, w = env.counts(s)
    mine, theirs = (b, w) if human_colour == BLACK else (w, b)
    print()
    if mine > theirs:
        print(st.good(st.bold(f"You win {mine}-{theirs}! 🎉" if st.color else f"You win {mine}-{theirs}!")))
        return 1
    if mine < theirs:
        print(st.bad(st.bold(f"{opp_name} wins {theirs}-{mine}.")))
        return -1
    print(st.bold(f"Draw, {mine}-{theirs}."))
    return 0


# --------------------------------------------------------------- tutorial
def pause(st):
    r = ask(st.dim("\n  [Enter] continue · [q] quit tutorial ")).strip().lower()
    if r in ("q", "quit"):
        raise KeyboardInterrupt


def say(text):
    print("\n" + textwrap.fill(textwrap.dedent(text).strip(), width=72, initial_indent="  ",
                               subsequent_indent="  "))


def tutorial(st, rng):
    env = OthelloEnv(6)
    try:
        print(st.bold("\n=== Othello tutorial ==="))
        say("""Othello is played on a square board by two players, Black and White.
            Each turn you place one disc of your colour. Black moves first. This
            tutorial uses a small 6x6 board; the classic game is 8x8.""")
        s = env.initial_state()
        print()
        print(render(env, s, st, show_legal=False))
        say("""The game starts with four discs in the centre. Squares are named by
            column letter and row number, like c2 or d5.""")
        pause(st)

        say("""RULE 1: CAPTURING. You must place your disc so that it sandwiches one
            or more of your opponent's discs in a straight line (across, down or
            diagonally) between the new disc and another of your discs. All
            sandwiched discs flip to your colour.""")
        print()
        print(render(env, s, st, show_legal=True))
        say(f"""The squares marked {st.legal()} are Black's legal moves. Let's say Black
            plays c2 (row 2, column c).""")
        pause(st)
        a = 1 * 6 + 2  # c2
        after = env.step(s, a)
        flipped = [sq for sq in range(36) if (s.white >> sq & 1) and (after.black >> sq & 1)]
        print()
        print(render(env, after, st, show_legal=False, last=a, highlight=flipped))
        say(f"""Black placed a disc on c2 and the White disc on
            {', '.join(env.action_name(q) for q in flipped)} was sandwiched, so it
            flipped to Black. One move can flip discs in several directions at once.""")
        pause(st)

        say("""RULE 2: PASSING. If you have no legal move you must pass, and your
            opponent moves again. You may not pass if you have a legal move.""")
        say("""RULE 3: THE END. The game ends when neither player can move (usually
            when the board is full). Whoever has more discs wins.""")
        pause(st)

        say("""STRATEGY 1: CORNERS ARE GOLD. A disc in a corner can never be
            flipped, and it lets you build more unflippable discs along the
            edges. In our experiments, a bot that ignored corners lost most of its
            strength.""")
        say("""STRATEGY 2: AVOID THE SQUARES NEXT TO AN EMPTY CORNER (especially the
            diagonal one, e.g. b2). Playing there often hands the corner to your
            opponent.""")
        say("""STRATEGY 3: DON'T GRAB DISCS EARLY. Counter-intuitively, having many
            discs mid-game is often bad: it gives your opponent more moves.
            Prefer moves that leave your opponent with FEW options (this is called
            mobility). A bot that only maximises its disc count was barely better
            than random in our tests.""")
        pause(st)
        say("""Time to practise! You'll play Black against the beginner bot on 6x6.
            Legal moves are marked, and a coach will tell you when the expert bot
            would have chosen differently. Type 'hint' for a suggestion.""")
        pause(st)
    except KeyboardInterrupt:
        print("\n  Tutorial ended. Run `othello --play` when you're ready.")
        return
    name, _, make = LEVELS[1]
    play_game(env, make(env, rng.randrange(10**9)), f"{name} bot", BLACK, st, hints=True, coach=True, rng=rng)
    print("\n  Nice! Next: `othello --play medium` or `othello --play 4 --size 8`.")


# -------------------------------------------------------------------- main
def parse_level(x):
    if x is None:
        return 3
    x = str(x).lower()
    if x.isdigit() and int(x) in LEVELS:
        return int(x)
    if x in LEVEL_NAMES:
        return LEVEL_NAMES[x]
    if x == FINAL_MODEL_NAME.lower():
        return x
    raise argparse.ArgumentTypeError(
        f"unknown level {x!r}; choose 1-5, one of {', '.join(LEVEL_NAMES)}, or {FINAL_MODEL_NAME.lower()}")


def build_parser():
    levels = "\n".join(f"  {k}  {name:<9} {desc}" for k, (name, desc, _) in LEVELS.items())
    p = argparse.ArgumentParser(
        prog="othello",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Play Othello in your terminal against bots or trained RL models.",
        epilog=f"levels:\n{levels}\n\nexamples:\n"
               "  othello --tutorial\n"
               "  othello --play                 # medium bot, 6x6\n"
               "  othello --play expert --size 8 --color white\n"
               "  othello --play oroletto        # the final trained model\n"
               "  othello --model results/classical/linear/seed0\n")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--play", nargs="?", const="3", metavar="LEVEL", type=parse_level,
                   help="play against a built-in bot (1-5 or name; default: medium), "
                        "or 'oroletto', the project's final trained model")
    g.add_argument("--model", metavar="PATH",
                   help="play against a trained model (run directory, model.pt or weights.json)")
    g.add_argument("--tutorial", action="store_true", help="learn the rules, then play a guided practice game")
    g.add_argument("--levels", action="store_true", help="list the difficulty levels")
    p.add_argument("--size", type=int, default=6, choices=[4, 6, 8, 10], help="board size (default: 6)")
    p.add_argument("--color", "--colour", dest="color", choices=["black", "white", "random"], default="black",
                   help="your colour; black moves first (default: black)")
    p.add_argument("--no-hints", action="store_true", help="don't mark legal moves on the board")
    p.add_argument("--coach", action="store_true", help="comment when the expert bot would have played differently")
    p.add_argument("--plain", action="store_true", help="ASCII board without colours (X/O)")
    p.add_argument("--seed", type=int, help="random seed for the bot")
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    color_ok = not a.plain and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None and os.name != "nt"
    st = Style(color_ok)
    rng = random.Random(a.seed)

    if a.levels:
        for k, (name, desc, _) in LEVELS.items():
            print(f"  {k}  {name:<9} {desc}")
        return 0
    if a.tutorial:
        tutorial(st, rng)
        return 0

    env = OthelloEnv(a.size)
    if a.model:
        from othello.models import load_model
        try:
            opponent = load_model(a.model, env)
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))
        opp_name = opponent.name
    elif a.play == FINAL_MODEL_NAME.lower():
        try:
            opponent = load_oroletto(env)
        except (ValueError, FileNotFoundError) as e:
            print(e, file=sys.stderr)
            return 1
        opp_name = FINAL_MODEL_NAME
    else:
        level = a.play if a.play is not None else 3
        name, _, make = LEVELS[level]
        opponent = make(env, rng.randrange(10**9))
        opp_name = f"{name} bot"

    colour = {"black": BLACK, "white": WHITE}.get(a.color, rng.choice([BLACK, WHITE]))
    tally = {1: 0, 0: 0, -1: 0}
    while True:
        r = play_game(env, opponent, opp_name, colour, st, hints=not a.no_hints, coach=a.coach, rng=rng)
        if r is None:
            break
        tally[r] += 1
        print(f"  Session: {tally[1]} won, {tally[0]} drawn, {tally[-1]} lost")
        again = ask("\nPlay again? [y/N/s = swap colours] ").strip().lower()
        if again in ("s", "swap"):
            colour = 1 - colour
        elif again not in ("y", "yes"):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
