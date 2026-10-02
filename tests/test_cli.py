import itertools

import pytest

from othello import cli


def scripted(monkeypatch, first, size=6, stop_after=None):
    squares = [c + str(r) for r in range(1, size + 1) for c in "abcdefghij"[:size]]
    feed = itertools.chain(first, itertools.cycle(squares))
    calls = {"n": 0}

    def fake_ask(prompt):
        calls["n"] += 1
        if "again" in prompt.lower():
            return "n"
        if stop_after and calls["n"] > stop_after:
            raise SystemExit(0)
        return next(feed)
    monkeypatch.setattr(cli, "ask", fake_ask)
    return calls


@pytest.mark.parametrize("level", ["1", "medium", "4"])
@pytest.mark.parametrize("colour", ["black", "white"])
def test_full_game_completes(monkeypatch, capsys, level, colour):
    scripted(monkeypatch, ["help", "moves", "hint", "zz9", "undo", "board"])
    assert cli.main(["--play", level, "--color", colour, "--plain", "--seed", "1", "--coach"]) == 0
    out = capsys.readouterr().out
    assert "Session:" in out and ("You win" in out or "wins" in out or "Draw" in out)


def test_quit(monkeypatch, capsys):
    scripted(monkeypatch, ["quit"])
    assert cli.main(["--play", "--plain"]) == 0
    assert "abandoned" in capsys.readouterr().out


def test_tutorial_runs_to_practice_game(monkeypatch, capsys):
    scripted(monkeypatch, [""] * 7)
    cli.main(["--tutorial", "--plain", "--seed", "0"])
    out = capsys.readouterr().out
    assert "CAPTURING" in out and "Session" not in out and "Next:" in out


def test_parse_move():
    env = cli.OthelloEnv(8)
    assert cli.parse_move(env, "a1") == 0
    assert cli.parse_move(env, "H8") == 63
    assert cli.parse_move(env, "4c") == 3 * 8 + 2
    assert cli.parse_move(env, "i1") is None
    assert cli.parse_move(env, "pass") == env.PASS


def test_bad_level_exits():
    with pytest.raises(SystemExit):
        cli.main(["--play", "9"])
