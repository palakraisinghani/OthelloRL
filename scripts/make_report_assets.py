"""Generate LaTeX tables and figures for report.tex from results/*.json.

  python scripts/make_report_assets.py
Writes report/tables/*.tex and report/figures/*.pdf. Never edit those by hand.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
TAB = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
INK, INK2, GRID, SERIES1 = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def load(p):
    return json.loads((RES / p).read_text())


def write(name, body):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(body)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def bold_if(s, cond):
    return rf"\textbf{{{s}}}" if cond else s


def score_ci(r):
    return rf"{pct(r['score'])} $\pm$ {pct(r['score_ci95'])}"


# --------------------------------------------------------------- throughput
def throughput():
    rows = load("baselines/throughput.json")
    best = max(r["games_per_sec"] for r in rows)
    lines = []
    for r in rows:
        gps = bold_if(f"{r['games_per_sec']:,.0f}", r["games_per_sec"] == best)
        lines.append(rf"{r['board_size']}$\times${r['board_size']} & {r['games']} & {gps} & "
                     rf"{r['plies_per_sec']:,.0f} & {r['mean_plies_per_game']:.1f} & {pct(r['black_score_random'])} \\")
    write("throughput.tex", "\n".join(lines) + "\n")


# ------------------------------------------------------------ weight tuning
def tuning():
    d = load("baselines/heuristic_tuning.json")
    rows, best = d["rows"], d["best"]
    find = lambda w: next(r for r in rows if r["weights"] == w)
    opp = ["disc-greedy", "hand-default", "random"]

    def line(label, r, bold_cols=(), removed=None):
        w = r["weights"]
        # the weight zeroed in a leave-one-out row is shown in red
        cells = [label] + [rf"\textcolor{{red!75!black}}{{\textbf{{{w[k]:g}}}}}" if k == removed else f"{w[k]:g}"
                           for k in ("disc", "mobility", "corner", "xsquare")]
        cells += [bold_if(pct(r["scores"][o]), o in bold_cols) for o in opp]
        cells.append(bold_if(pct(r["mean_score"]), "mean" in bold_cols))
        return " & ".join(cells) + r" \\"

    shown = [(f"Rank {i}" + (" (selected)" if i == 1 else ""), r) for i, r in enumerate(rows[:5], 1)]
    shown.append(None)
    for k in ["disc", "mobility", "corner", "xsquare"]:
        w = dict(best)
        if w[k] != 0:
            w[k] = 0
            shown.append((f"Selected, no {k.replace('xsquare', 'X-square')}", find(w), k))
    shown.append(None)
    shown.append(("Disc count only", find({"disc": 1, "mobility": 0, "corner": 0, "xsquare": 0})))
    shown.append(("Hand default", find({"disc": 1, "mobility": 5, "corner": 25, "xsquare": 0})))
    rs = [x[1] for x in shown if x]
    col_best = {o: max(r["scores"][o] for r in rs) for o in opp}
    col_best["mean"] = max(r["mean_score"] for r in rs)
    out = []
    for x in shown:
        if x is None:
            out.append(r"\midrule")
            continue
        label, r = x[0], x[1]
        removed = x[2] if len(x) > 2 else None
        bc = [o for o in opp if r["scores"][o] == col_best[o]] + (["mean"] if r["mean_score"] == col_best["mean"] else [])
        out.append(line(label, r, bc, removed))
    write("tuning.tex", "\n".join(out) + "\n")
    write("tuning_meta.tex", rf"\newcommand{{\NumCandidates}}{{{len(rows)}}}" + "\n")
    # the selected weights, used by every benchmark in the report
    desc = {"disc": "own $-$ opponent discs", "mobility": "own $-$ opponent legal moves",
            "corner": "own $-$ opponent corner discs",
            "xsquare": "own $-$ opponent discs on an X-square next to an empty corner"}
    label = {"disc": "Disc count", "mobility": "Mobility", "corner": "Corners", "xsquare": "X-squares"}
    write("heuristic_weights.tex", "\n".join(
        rf"{label[k]} & {desc[k]} & {best[k]:g} \\" for k in ("corner", "mobility", "disc", "xsquare")) + "\n")


# ----------------------------------------------------------- minimax depth
def depth():
    rows = load("baselines/minimax_depth.json")
    best = max(r["score"] for r in rows)
    out = []
    prev = None
    for r in rows:
        ebf = f"{r['nodes_per_move'] / prev:.1f}" if prev else "--"
        prev = r["nodes_per_move"]
        out.append(rf"{r['depth']} & {bold_if(score_ci(r), r['score'] == best)} & {r['wins']}/{r['draws']}/{r['losses']} & "
                   rf"{r['mean_margin']:+.1f} & {r['nodes_per_move']:.0f} & {ebf} & {1000 * r['sec_per_game_both_sides']:.0f} \\")
    write("minimax_depth.tex", "\n".join(out) + "\n")

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = [r["nodes_per_move"] for r in rows]
    y = [100 * r["score"] for r in rows]
    e = [100 * r["score_ci95"] for r in rows]
    ax.errorbar(x, y, yerr=e, color=SERIES1, lw=2, marker="o", ms=6, capsize=3,
                markeredgecolor="white", markeredgewidth=1.5, elinewidth=1)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(f"d={r['depth']}", (xi, yi), textcoords="offset points", xytext=(7, -4),
                    fontsize=8, color=INK2)
    ax.axhline(50, color=INK2, lw=0.8, ls=":")
    ax.text(x[-1], 51, "parity", fontsize=7, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("search nodes per move (log scale)")
    ax.set_ylabel("score vs heuristic bot (%)")
    ax.set_ylim(45, 100)
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "minimax_depth.pdf")
    plt.close(fig)


# -------------------------------------------------------------------- gate
# Gate on score (draws = 1/2). The 95% bar was revised to 90% after the
# diagnostics in gate_diag.json / gate_openings.json (see report).
THRESH = {("heuristic", "random"): 0.90, ("minimax-d4", "heuristic"): 0.80}


def gate():
    g = load("baselines/gate.json")
    out = []
    for n, v in g["sizes"].items():
        first = True
        for m in v["matches"]:
            key = (m["a"], m["b"])
            th = THRESH.get(key)
            if th is None:
                verdict = "--"
            else:
                verdict = r"PASS" if m["score"] >= th else r"\textbf{FAIL}"
                verdict = rf"$\geq${pct(th, 0)} \;{verdict}"
            size = rf"{n}$\times${n}" if first else ""
            first = False
            out.append(rf"{size} & {m['a']} vs {m['b']} & {m['games']} & {m['wins']}/{m['draws']}/{m['losses']} & "
                       rf"{pct(m['win_rate'])} & {score_ci(m)} & {m['mean_margin']:+.1f} & {verdict} \\")
        out.append(r"\midrule")
    write("gate.tex", "\n".join(out[:-1]) + "\n")
    gate_bars(g)


# ordinal blue ramp (steps 250/450/650): board size is ordered, not categorical
SIZE_RAMP = {"6": "#86b6ef", "8": "#2a78d6", "10": "#104281"}


def gate_bars(g):
    import numpy as np
    order = [("heuristic", "random"), ("minimax-d4", "heuristic"), ("minimax-d4", "random")]
    labels = ["heuristic\nvs random", "minimax-d4\nvs heuristic", "minimax-d4\nvs random"]
    sizes = list(g["sizes"])
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(order))
    w = 0.26
    for j, n in enumerate(sizes):
        ms = {(m["a"], m["b"]): m for m in g["sizes"][n]["matches"]}
        vals = [100 * ms[k]["score"] for k in order]
        err = [100 * ms[k]["score_ci95"] for k in order]
        xs = x + (j - 1) * (w + 0.02)
        ax.bar(xs, vals, w, color=SIZE_RAMP[n], edgecolor="white", linewidth=1,
               label=f"{n}$\\times${n}  ({ms[order[0]]['games']} games/match-up)")
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
        for xi, v, e in zip(xs, vals, err):
            ax.text(xi, v + e + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    first = True
    for i, k in enumerate(order):
        th = THRESH.get(k)
        if th:
            ax.hlines(100 * th, i - 0.45, i + 0.45, colors=INK, linestyles="--", lw=1.0,
                      label="gate threshold" if first else None)
            first = False
    ax.axhline(50, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score of the first-named bot (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    _style_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "bots_head_to_head.pdf")
    plt.close(fig)


def diag():
    rows = load("baselines/gate_diag.json")
    by = {}
    for r in rows:
        by.setdefault(r["agent"], {})[r["board_size"]] = r
    best = {s: max(r[s]["win_rate"] for r in by.values()) for s in (6, 8)}
    out = []
    for agent, r in by.items():
        cells = [agent.replace("eps", r"$\varepsilon$")]
        for s in (6, 8):
            cells += [bold_if(pct(r[s]["win_rate"]), r[s]["win_rate"] == best[s]), f"{r[s]['mean_margin']:+.1f}"]
        out.append(" & ".join(cells) + r" \\")
    write("gate_diag.tex", "\n".join(out) + "\n")


def openings():
    rows = load("baselines/gate_openings.json")
    out = []
    for p in sorted({r["opening_plies"] for r in rows}):
        r5, r0 = [next(r for r in rows if r["opening_plies"] == p and r["epsilon"] == e) for e in (0.05, 0.0)]
        out.append(rf"{p} & {r5['distinct_openings']} & {pct(r5['win_rate'])} & {score_ci(r5)} & {pct(r0['win_rate'])} & {score_ci(r0)} \\")
    write("gate_openings.tex", "\n".join(out) + "\n")


def meta():
    n = (RES / "baselines/pytest.txt").read_text().split()[0]
    write("tests_meta.tex", rf"\newcommand{{\NumTests}}{{{n}}}" + "\n")
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/classical.yaml").read_text())
    write("classical_meta.tex", rf"\newcommand{{\TrainGames}}{{{cfg['train_games']:,}}}".replace(",", "{,}") + "\n")


# ------------------------------------------------------------ classical RL
MAIN = [("q-learning", "Q-learning"), ("sarsa", "SARSA"), ("mc-control", "MC control"),
        ("linear", "Linear FA"), ("dqn", "DQN")]
ABLATION = [("linear", "All features, win/loss reward"), ("linear-nodisc", "No disc feature"),
            ("linear-disconly", "Disc feature only"), ("linear-discreward", "All features, disc-margin reward")]
OPPS = [("random", "random"), ("heuristic", "heuristic"), ("minimax-d4", "minimax-d4")]
# categorical slots 1-5 of the reference palette (fixed order, never cycled)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _runs(name):
    out = []
    for d in sorted((RES / "classical" / name).glob("seed*")):
        if (d / "final.json").exists():
            out.append({"final": json.loads((d / "final.json").read_text()), "dir": d})
    return out


def _curve(d):
    import csv
    with open(d / "curve.csv") as f:
        return list(csv.DictReader(f))


def _ms(vals):
    import statistics
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def _style_axes(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def classical():
    import numpy as np
    have = {k: _runs(k) for k, _ in MAIN + ABLATION}
    missing = [k for k, v in have.items() if not v]
    if missing:
        print("classical: no finished runs for", missing)
    main = [(k, lab) for k, lab in MAIN if have[k]]
    if not main:
        return

    # ---- results table
    rows = []
    best = {o: max(_ms([r["final"][o]["score"] for r in have[k]])[0] for k, _ in main) for o, _ in OPPS}
    for k, lab in main:
        cells = [lab, str(len(have[k]))]
        for o, _ in OPPS:
            m, sd = _ms([r["final"][o]["score"] for r in have[k]])
            cells.append(bold_if(rf"{pct(m)} $\pm$ {pct(sd)}", m == best[o]))
        wr, _ = _ms([r["final"]["heuristic"]["win_rate"] for r in have[k]])
        cells.append(pct(wr))
        rows.append(" & ".join(cells) + r" \\")
    # reference row: the heuristic bot itself, from the rule-based round robin
    g6 = {(m["a"], m["b"]): m for m in load("baselines/gate.json")["sizes"]["6"]["matches"]}
    h_r = g6[("heuristic", "random")]
    h_m = g6[("minimax-d4", "heuristic")]
    rows.append(r"\midrule")
    rows.append(rf"\textit{{Heuristic bot (reference)}} & -- & {pct(h_r['score'])} & -- & "
                rf"{pct(1 - h_m['score'])} & -- \\")
    write("classical.tex", "\n".join(rows) + "\n")

    # ---- grouped bar chart: every method vs the 3 rule-based bots
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})

    def bars(groups, fname, width_in):
        fig, ax = plt.subplots(figsize=(width_in, 3.0))
        x = np.arange(len(groups))
        w = 0.26
        for j, (o, olab) in enumerate(OPPS):
            means, lo, hi = [], [], []
            for k, _ in groups:
                v = [100 * r["final"][o]["score"] for r in have[k]]
                m = float(np.mean(v))
                means.append(m)
                lo.append(m - min(v))
                hi.append(max(v) - m)
            xs = x + (j - 1) * (w + 0.02)
            ax.bar(xs, means, w, color=CAT[j], label=f"vs {olab}", edgecolor="white", linewidth=1)
            ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
            for xi, m in zip(xs, means):
                ax.text(xi, m + 2.5, f"{m:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.axhline(50, color=INK, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in groups])
        ax.set_ylabel("final score (%)")
        ax.set_ylim(0, 108)
        _style_axes(ax)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)

    bars(main, "classical_bars.pdf", 6.2)
    abl = [(k, lab) for k, lab in ABLATION if have[k]]
    short = {"linear": "all features", "linear-nodisc": "no disc", "linear-disconly": "disc only",
             "linear-discreward": "disc-margin\nreward"}
    if len(abl) > 1:
        bars([(k, short[k]) for k, _ in abl], "classical_ablation_bars.pdf", 5.2)

    # ---- learning curves vs heuristic
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for c, (k, lab) in zip(CAT, main):
        curves = [_curve(r["dir"]) for r in have[k]]
        L = min(len(cv) for cv in curves)
        g = np.array([float(curves[0][i]["games"]) for i in range(L)]) / 1000
        y = np.array([[100 * float(cv[i]["score"]) for i in range(L)] for cv in curves])
        ax.fill_between(g, y.min(0), y.max(0), color=c, alpha=0.15, lw=0)
        ax.plot(g, y.mean(0), color=c, lw=2, label=lab)
    ax.axhline(50, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("training games (thousands)")
    ax.set_ylabel("score vs heuristic (%)")
    ax.set_ylim(0, 100)
    _style_axes(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "classical_curves.pdf")
    plt.close(fig)

    # ---- tabular diagnostics
    rows = []
    for k, lab in MAIN[:3]:
        if not have[k]:
            continue
        last = [_curve(r["dir"])[-1] for r in have[k]]
        first_eval = [_curve(r["dir"])[1] for r in have[k] if len(_curve(r["dir"])) > 1]
        states, _ = _ms([float(l["table_states"]) for l in last])
        cov, _ = _ms([r["final"]["heuristic"]["coverage"] for r in have[k]])
        covr, _ = _ms([r["final"]["random"]["coverage"] for r in have[k]])
        var_end, _ = _ms([float(l["update_err_var"]) for l in last])
        var_start, _ = _ms([float(l["update_err_var"]) for l in first_eval]) if first_eval else (float("nan"), 0)
        rows.append(rf"{lab} & {states / 1e6:.2f} & {pct(cov)} & {pct(covr)} & {var_start:.3f} & {var_end:.3f} \\")
    write("classical_tabular.tex", "\n".join(rows) + "\n")

    # ---- disc-count ablation
    rows = []
    for k, lab in abl:
        cells = [lab]
        for o, _ in OPPS:
            m, sd = _ms([r["final"][o]["score"] for r in have[k]])
            cells.append(rf"{pct(m)} $\pm$ {pct(sd)}")
        ws = [json.loads((r["dir"] / "weights.json").read_text()) for r in have[k]]
        for f in ("disc", "mobility", "corner"):
            v = [w[f] for w in ws if f in w]
            cells.append(f"{_ms(v)[0]:+.2f}" if v else "--")
        rows.append(" & ".join(cells) + r" \\")
    write("classical_ablation.tex", "\n".join(rows) + "\n")


# ------------------------------------------------------------- self-play
SP_RUNS = [("selfplay/dqn-plus", "DQN+"), ("selfplay/ppo", "PPO"),
           ("alphazero", "AlphaZero (net only)"), ("alphazero", "AlphaZero")]


def _sp_runs(path):
    root = RES / path
    out = []
    for d in sorted(root.glob("seed*")):
        if (d / "final.json").exists():
            out.append({"final": json.loads((d / "final.json").read_text()), "dir": d})
    return out


def _sp_score(run, opp, label):
    key = f"policy_only/{opp}" if label == "AlphaZero (net only)" else opp
    return run["final"][key]["score"]


def _games_to(curves, thr, col="score"):
    """Mean over seeds of the first evaluated game count at which `col` >= thr ('--' if any seed never does)."""
    firsts = []
    for cv in curves:
        hit = [float(r["games"]) for r in cv if float(r[col]) >= thr and float(r["games"]) > 0]
        if not hit:
            return None
        firsts.append(hit[0])
    return sum(firsts) / len(firsts)


def selfplay():
    import numpy as np
    groups = [("classical/dqn", "DQN (classical)")] + SP_RUNS
    have = {lab: _sp_runs(path) for path, lab in groups}
    if not all(have.values()):
        print("selfplay: missing runs", [lab for lab, v in have.items() if not v])
        return
    # ---- comparison table
    best = {o: max(_ms([_sp_score(r, o, lab) for r in have[lab]])[0] for _, lab in groups) for o, _ in OPPS}
    rows = []
    for path, lab in groups:
        runs = have[lab]
        cells = [lab, str(len(runs))]
        for o, _ in OPPS:
            m, sd = _ms([_sp_score(r, o, lab) for r in runs])
            cells.append(bold_if(rf"{pct(m)} $\pm$ {pct(sd)}", m == best[o]))
        col = "policy_score" if lab == "AlphaZero (net only)" else "score"
        curves = [_curve(r["dir"]) for r in runs]
        first_eval = min(float(r["games"]) for r in curves[0] if float(r["games"]) > 0)
        for thr in (0.6, 0.8):
            g = _games_to(curves, thr, col)
            # reached at the first evaluation point: only an upper bound is known
            cells.append("--" if g is None else (rf"$\le${g / 1000:.0f}k" if g <= first_eval else f"{g / 1000:.0f}k"))
        games = float(curves[0][-1]["games"])
        hours, _ = _ms([r["final"]["train_seconds"] / 3600 for r in runs])
        cells += [f"{games / 1000:.0f}k", "" if lab == "AlphaZero (net only)" else f"{hours:.1f}"]
        rows.append(" & ".join(cells) + r" \\")
        if lab == "DQN (classical)":
            rows.append(r"\midrule")
    write("selfplay.tex", "\n".join(rows) + "\n")

    # ---- grouped bars vs the three bots
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(groups))
    w = 0.26
    for j, (o, olab) in enumerate(OPPS):
        means, lo, hi = [], [], []
        for _, lab in groups:
            v = [100 * _sp_score(r, o, lab) for r in have[lab]]
            m = float(np.mean(v))
            means.append(m)
            lo.append(m - min(v))
            hi.append(max(v) - m)
        xs = x + (j - 1) * (w + 0.02)
        ax.bar(xs, means, w, color=CAT[j], label=f"vs {olab}", edgecolor="white", linewidth=1)
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
        for xi, m, h in zip(xs, means, hi):
            ax.text(xi, m + h + 1.5, f"{m:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.axhline(50, color=INK, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace(" (", "\n(") for _, lab in groups])
    ax.set_ylabel("final score (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    _style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "selfplay_bars.pdf")
    plt.close(fig)

    # ---- learning curves vs heuristic
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    series = [("DQN (classical)", "score", CAT[4], "-"), ("DQN+", "score", CAT[0], "-"),
              ("PPO", "score", CAT[1], "-"), ("AlphaZero", "score", CAT[2], "-"),
              ("AlphaZero (net only)", "policy_score", CAT[2], "--")]
    for lab, col, c, ls in series:
        curves = [_curve(r["dir"]) for r in have[lab]]
        L = min(len(cv) for cv in curves)
        g = np.array([float(curves[0][i]["games"]) for i in range(L)]) / 1000
        y = np.array([[100 * float(cv[i][col]) for i in range(L)] for cv in curves])
        ax.fill_between(g, y.min(0), y.max(0), color=c, alpha=0.15, lw=0)
        ax.plot(g, y.mean(0), color=c, lw=2, ls=ls, label=lab)
    ax.axhline(50, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("training games (thousands)")
    ax.set_ylabel("score vs heuristic (%)")
    ax.set_ylim(0, 100)
    _style_axes(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "selfplay_curves.pdf")
    plt.close(fig)

    # ---- solved-6x6 check
    sc = RES / "alphazero" / "solved_check.json"
    if sc.exists():
        d = json.loads(sc.read_text())
        rows = []
        for sims in sorted({g["sims"] for g in d["games"]}):
            gs = [g for g in d["games"] if g["sims"] == sims]
            res = ", ".join(sorted({f"{g['black']}--{g['white']}" for g in gs}))
            win = {("W" if g["white"] > g["black"] else "B" if g["black"] > g["white"] else "D") for g in gs}
            rows.append(rf"{sims} & {len(gs)} & {res} & {'White' if win == {'W'} else 'mixed'} \\")
        rows.append(r"\midrule")
        rows.append(r"\textit{Perfect play} & -- & 16--20 & White \\")
        write("solved.tex", "\n".join(rows) + "\n")
        write("solved_meta.tex", rf"\newcommand{{\StartValue}}{{{d['start_value_black']:+.2f}}}" + "\n"
              + rf"\newcommand{{\OrolettoRun}}{{{Path(d['model']).name}}}" + "\n")


# ---------------------------------------------------------------- league (Phase 3)
LG_MIXES = ["p0", "p25", "p50", "p75", "p100"]
LG_P = {"p0": 0.0, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p100": 1.0}
# ordinal ramp for the ordered mix parameter p (blue steps 250..650)
P_RAMP = {"p0": "#86b6ef", "p25": "#5598e7", "p50": "#2a78d6", "p75": "#1c5cab", "p100": "#104281"}
HELD = [("p0", "self-play DQN (p=0)"), ("minimax-d2", "minimax-d2"), ("minimax-d3", "minimax-d3"),
        ("ppo", "Phase 2 PPO"), ("dqn+", "Phase 2 DQN+")]
INLG = [("random", "random"), ("heuristic", "heuristic"), ("minimax-d4", "minimax-d4")]
# colour follows the opponent across the whole report: random/heuristic/minimax-d4 keep
# slots 1-3 (as in every earlier chart); held-out opponents take slots 4-8
OPP_COLOR = {"random": "#2a78d6", "heuristic": "#eb6834", "minimax-d4": "#1baf7a", "p0": "#eda100",
             "minimax-d2": "#e87ba4", "minimax-d3": "#008300", "ppo": "#4a3aa7", "dqn+": "#e34948"}


def _lg_score(r, key):
    """Model result dict -> score vs a panel opponent (averaging the seed-level entries)."""
    if key in ("p0", "ppo", "dqn+"):
        pref = {"p0": "p0-s", "ppo": "ppo-s", "dqn+": "dqn+-s"}[key]
        v = [x["score"] for k, x in r.items() if k.startswith(pref)]
        return sum(v) / len(v)
    return r[key]["score"]


def _boot_vs_p0(ev, mix, n=10_000, seed=0):
    """Mean score of `mix` vs the p0 models over seed pairs, with a seed-level bootstrap 95% CI."""
    import numpy as np
    rng = np.random.default_rng(seed)
    S = [0, 1, 2]
    M = {(a, b): ev[f"{mix}/seed{a}"][f"p0-s{b}"]["score"] for a in S for b in S
         if f"p0-s{b}" in ev[f"{mix}/seed{a}"]}
    mean = sum(M.values()) / len(M)
    boots = []
    for _ in range(n):
        A, B = rng.choice(S, 3), rng.choice(S, 3)
        v = [M[(a, b)] for a in A for b in B if (a, b) in M]
        if v:
            boots.append(sum(v) / len(v))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, lo, hi


def league():
    import numpy as np
    ev_path = RES / "league" / "dqn_eval.json"
    if not ev_path.exists():
        print("league: no evaluation yet")
        return
    ev = json.loads(ev_path.read_text())
    if not all(f"{m}/seed{s}" in ev for m in LG_MIXES for s in (0, 1, 2)):
        print("league: evaluation incomplete")
        return
    cfg = yaml_load(ROOT / "configs/league_dqn.yaml")

    # ---- league members table (strength from earlier phases: score vs heuristic)
    g6 = {(m["a"], m["b"]): m for m in load("baselines/gate.json")["sizes"]["6"]["matches"]}
    ref = {"random": 1 - g6[("heuristic", "random")]["score"], "heuristic": None,
           "minimax-d4": g6[("minimax-d4", "heuristic")]["score"]}
    kinds = {"random": "rule-based", "heuristic": "rule-based", "minimax-d4": "rule-based"}
    rows = []
    for m in cfg["league_members"]:
        nm = m["name"]
        if "path" in m:
            f = json.loads((ROOT / m["path"] / "final.json").read_text())
            sc, kind = f["heuristic"]["score"], "traditional RL (Phase 1)"
        else:
            sc, kind = ref[nm], kinds[nm]
        rows.append(rf"{nm} & {kind} & {'--' if sc is None else pct(sc)} \\")
    rows.append(r"own snapshots & frozen copies of the learner (every 5k games, $\le$20) & varies \\")
    write("league_members.tex", "\n".join(rows) + "\n")

    # ---- results table
    boot = {m: _boot_vs_p0(ev, m) for m in LG_MIXES}
    per = {m: {k: [_lg_score(ev[f"{m}/seed{s}"], k) for s in (0, 1, 2)] for k, _ in HELD[1:] + INLG}
           for m in LG_MIXES}
    best = {k: max(np.mean(per[m][k]) for m in LG_MIXES) for k, _ in HELD[1:] + INLG}
    best_b = max(boot[m][0] for m in LG_MIXES)
    rows = []
    for m in LG_MIXES:
        mean, lo, hi = boot[m]
        cells = [rf"{LG_P[m]:g}", bold_if(rf"{pct(mean)} [{pct(lo)}, {pct(hi)}]", mean == best_b)]
        for k, _ in HELD[1:] + INLG:
            v = per[m][k]
            cells.append(bold_if(rf"{pct(np.mean(v))} $\pm$ {pct(np.std(v, ddof=1))}", np.mean(v) == best[k]))
        rows.append(" & ".join(cells) + r" \\")
    write("league_results.tex", "\n".join(rows) + "\n")

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    xs = [LG_P[m] for m in LG_MIXES]

    # ---- score vs p: held-out (left) and in-league (right)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, group, title in ((axes[0], HELD, "held-out opponents"), (axes[1], INLG, "in-league opponents")):
        for j, (k, lab) in enumerate(group):
            if k == "p0":
                y = [100 * boot[m][0] for m in LG_MIXES]
                lo = [100 * (boot[m][0] - boot[m][1]) for m in LG_MIXES]
                hi = [100 * (boot[m][2] - boot[m][0]) for m in LG_MIXES]
            else:
                y = [100 * np.mean(per[m][k]) for m in LG_MIXES]
                lo = [100 * (np.mean(per[m][k]) - min(per[m][k])) for m in LG_MIXES]
                hi = [100 * (max(per[m][k]) - np.mean(per[m][k])) for m in LG_MIXES]
            c = OPP_COLOR[k]
            ax.errorbar(xs, y, yerr=[lo, hi], color=c, lw=2, marker="o", ms=5, capsize=2,
                        markeredgecolor="white", markeredgewidth=1, label=lab)
        ax.axhline(50, color=INK, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("league share p")
        ax.set_xticks(xs)
        ax.set_ylim(0, 100)
        _style_axes(ax)
        if group is HELD:
            ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
        else:   # empty band between the heuristic (~55%) and minimax-d4 (~15%) lines
            ax.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.35))
    axes[0].set_ylabel("score (%)")
    fig.tight_layout()
    fig.savefig(FIG / "league_mix_curve.pdf")
    plt.close(fig)

    # ---- best mix vs p0 bars
    best_mix = max(LG_MIXES[1:], key=lambda m: boot[m][0])
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    keys = [(k, lab) for k, lab in HELD[1:] + INLG]
    x = np.arange(len(keys))
    w = 0.38
    for j, (m, c) in enumerate((("p0", P_RAMP["p0"]), (best_mix, P_RAMP["p100"]))):
        means = [100 * np.mean(per[m][k]) for k, _ in keys]
        lo = [100 * (np.mean(per[m][k]) - min(per[m][k])) for k, _ in keys]
        hi = [100 * (max(per[m][k]) - np.mean(per[m][k])) for k, _ in keys]
        xx = x + (j - 0.5) * (w + 0.02)
        ax.bar(xx, means, w, color=c, edgecolor="white", linewidth=1, label=f"p = {LG_P[m]:g}")
        ax.errorbar(xx, means, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
        for xi, mval, h in zip(xx, means, hi):
            ax.text(xi, mval + h + 1.5, f"{mval:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.axvline(len(HELD) - 1.5, color=INK2, lw=0.8, ls="--")
    ax.text(len(HELD) - 1.45, 103, "in-league →", fontsize=7, color=INK2)
    ax.axhline(50, color=INK, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace("Phase 2 ", "P2 ") for _, lab in keys], fontsize=7.5)
    ax.set_ylabel("final score (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    _style_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "league_best_vs_p0.pdf")
    plt.close(fig)
    write("league_meta.tex", rf"\newcommand{{\LeagueBestMix}}{{{LG_P[best_mix]:g}}}" + "\n")

    # ---- head-to-head matrix (row mix vs column mix)
    hp = RES / "league" / "dqn_h2h_matrix.json"
    if hp.exists():
        H = json.loads(hp.read_text())
        mat = np.full((5, 5), np.nan)
        for i, ma in enumerate(LG_MIXES):
            for j, mb in enumerate(LG_MIXES):
                if i == j:
                    continue
                vals = []
                for k, v in H.items():
                    a, b = k.split("|")
                    if a.startswith(ma + "/") and b.startswith(mb + "/"):
                        vals.append(v["score"])
                    elif a.startswith(mb + "/") and b.startswith(ma + "/"):
                        vals.append(1 - v["score"])
                if vals:
                    mat[i, j] = 100 * np.mean(vals)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        im = ax.imshow(mat, cmap=cmap, vmin=20, vmax=80)
        for i in range(5):
            for j in range(5):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, color=INK)
        lab = [f"{LG_P[m]:g}" for m in LG_MIXES]
        ax.set_xticks(range(5), lab)
        ax.set_yticks(range(5), lab)
        ax.set_xlabel("opponent: league share p")
        ax.set_ylabel("player: league share p")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("player's score (%)", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "league_h2h.pdf")
        plt.close(fig)

    # ---- checkpoint curves vs minimax-d3 (H4) + per-mix table
    cp = RES / "league" / "dqn_ckpt_curves.json"
    if cp.exists():
        C = json.loads(cp.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
        h4 = []
        for m in LG_MIXES:
            for ax, opp in zip(axes, ("minimax-d3", "ppo-s0")):
                by = {}
                for k, v in C.items():
                    if k.startswith(m + "/"):
                        by.setdefault(v["games"], []).append(100 * v[opp]["score"])
                g = sorted(by)
                ax.plot([x / 1000 for x in g], [np.mean(by[x]) for x in g], color=P_RAMP[m], lw=2,
                        marker="o", ms=3, label=f"p = {LG_P[m]:g}")
            # buffer composition at the end of training (mean over seeds)
            shares = []
            for sd in (0, 1, 2):
                rowsc = _curve(RES / "league" / "dqn" / m / f"seed{sd}")
                shares.append(float(rowsc[-1].get("buffer_league_share", "nan")))
            early = [100 * v["minimax-d3"]["score"] for k, v in C.items() if k.startswith(m + "/") and v["games"] == 50000]
            last = [100 * v["minimax-d3"]["score"] for k, v in C.items() if k.startswith(m + "/") and v["games"] == max(x["games"] for x in C.values())]
            h4.append(rf"{LG_P[m]:g} & {100 * np.mean(shares):.0f} & {np.mean(early):.1f} & {np.mean(last):.1f} \\")
        for ax, t in zip(axes, ("vs minimax-d3 (held-out)", "vs Phase 2 PPO seed 0 (held-out)")):
            ax.set_title(t, fontsize=9, color=INK)
            ax.set_xlabel("training games (thousands)")
            ax.axhline(50, color=INK, lw=0.8, ls=":")
            ax.set_ylim(0, 100)
            _style_axes(ax)
        axes[0].set_ylabel("score (%)")
        axes[1].legend(frameon=False, fontsize=6.5, loc="upper left")
        fig.tight_layout()
        fig.savefig(FIG / "league_ckpt_curves.pdf")
        plt.close(fig)
        write("league_h4.tex", "\n".join(h4) + "\n")

    # ---- training win rate vs each league member over time (p = 0.5, mean over seeds)
    names = ["self"] + [m["name"] for m in cfg["league_members"]] + ["snapshots"]
    curves = [_curve(RES / "league" / "dqn" / "p50" / f"seed{sd}") for sd in (0, 1, 2)]
    L = min(len(c) for c in curves)
    mat = np.full((len(names), L - 1), np.nan)
    for i, nm in enumerate(names):
        for t in range(1, L):
            vals = [float(c[t][f"vs_{nm}_score"]) for c in curves if c[t][f"vs_{nm}_score"] not in ("", "nan")]
            if vals:
                mat[i, t - 1] = 100 * np.mean(vals)
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    im = ax.imshow(mat, aspect="auto", cmap=seq, vmin=0, vmax=100,
                   extent=[2.5, 5 * (L - 1) + 2.5, len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.set_xlabel("training games (thousands)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("training score (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "league_opponent_curves.pdf")
    plt.close(fig)


# ------------------------------------------------- fixed-opponent ceiling (Step 2.9)
FO_GROUPS = [("dqn-random", "vs-random", "DQN trained vs random"),
             ("dqn-heuristic", "vs-heuristic", "DQN trained vs heuristic"),
             ("dqn-minimax-d3", "vs-minimax-d3", "DQN trained vs minimax-d3"),
             ("dqn-selfplay", None, "DQN trained by self-play"),
             ("dqn-classical", "classical", "classical DQN (Phase 1, vs heuristic)")]
FO_BOTS = [("random", "random"), ("heuristic", "heuristic"), ("minimax-d2", "minimax-d2"),
           ("minimax-d3", "minimax-d3"), ("minimax-d4", "minimax-d4")]
FO_TRAINER = {"dqn-random": "random", "dqn-heuristic": "heuristic", "dqn-minimax-d3": "minimax-d3",
              "dqn-classical": "heuristic"}


def _fo_dir(g, run, seed):
    if run is None:
        return RES / "league/dqn/p0" / f"seed{seed}"
    if run == "classical":
        return RES / "classical/dqn" / f"seed{seed}"
    return RES / "fixed_opponent" / run / f"seed{seed}"


def _fo_final(g, run, seed):
    """final.json of a learner; the Phase 1 classical DQN gets its d2/d3 results from classical_panel.json."""
    f = json.loads((_fo_dir(g, run, seed) / "final.json").read_text())
    if run == "classical":
        pn = json.loads((RES / "fixed_opponent" / "classical_panel.json").read_text())
        for b in ("minimax-d2", "minimax-d3"):
            f[b] = pn[f"seed{seed}/{b}"]
    return f


def fixed_opponent():
    import numpy as np
    hp = RES / "fixed_opponent" / "h2h.json"
    fins = {}
    for g, run, _ in FO_GROUPS:
        fs = [_fo_dir(g, run, s) / "final.json" for s in (0, 1, 2)]
        if not all(f.exists() for f in fs):
            print("fixed_opponent: runs incomplete")
            return
        if run == "classical" and not (RES / "fixed_opponent" / "classical_panel.json").exists():
            print("fixed_opponent: classical panel missing")
            return
        fins[g] = [_fo_final(g, run, sd) for sd in (0, 1, 2)]
    # ---- panel table: each learner vs every bot (its own trainer marked)
    best = {b: max(np.mean([f[b]["score"] for f in fins[g]]) for g, _, _ in FO_GROUPS) for b, _ in FO_BOTS}
    rows = []
    for g, run, lab in FO_GROUPS:
        cells = [lab]
        for b, _ in FO_BOTS:
            v = [f[b]["score"] for f in fins[g]]
            c = rf"{pct(np.mean(v))} $\pm$ {pct(np.std(v, ddof=1))}"
            c = bold_if(c, np.mean(v) == best[b])
            if FO_TRAINER.get(g) == b:
                c = rf"\underline{{{c}}}$^\dagger$"
            cells.append(c)
        rows.append(" & ".join(cells) + r" \\")
        if g == "dqn-minimax-d3":
            rows.append(r"\midrule")
    write("fo_panel.tex", "\n".join(rows) + "\n")

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    # ---- bars: learners x bots
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(FO_GROUPS))
    w = 0.16
    for j, (b, blab) in enumerate(FO_BOTS):
        means = [100 * np.mean([f[b]["score"] for f in fins[g]]) for g, _, _ in FO_GROUPS]
        lo = [m - 100 * min(f[b]["score"] for f in fins[g]) for m, (g, _, _) in zip(means, FO_GROUPS)]
        hi = [100 * max(f[b]["score"] for f in fins[g]) - m for m, (g, _, _) in zip(means, FO_GROUPS)]
        xs = x + (j - 2) * (w + 0.01)
        ax.bar(xs, means, w, color=OPP_COLOR[b], edgecolor="white", linewidth=0.8, label=f"vs {blab}")
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
    ax.axhline(50, color=INK, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace("DQN trained ", "trained\n").replace("classical DQN (Phase 1, vs heuristic)",
                        "classical DQN\n(Phase 1)") for _, _, lab in FO_GROUPS], fontsize=7.5)
    ax.set_ylabel("final score (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    _style_axes(ax)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fo_bars.pdf")
    plt.close(fig)

    # ---- learning curves vs minimax-d2 (held out for all three)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ramp = {"dqn-random": "#86b6ef", "dqn-heuristic": "#2a78d6", "dqn-minimax-d3": "#104281"}
    for g, run, lab in FO_GROUPS[:3]:
        cv = [_curve(_fo_dir(g, run, s)) for s in (0, 1, 2)]
        L = min(len(c) for c in cv)
        gx = np.array([float(cv[0][i]["games"]) for i in range(L)]) / 1000
        y = np.array([[100 * float(c[i]["score"]) for i in range(L)] for c in cv])
        ax.fill_between(gx, y.min(0), y.max(0), color=ramp[g], alpha=0.15, lw=0)
        ax.plot(gx, y.mean(0), color=ramp[g], lw=2, label=lab)
    sp = 100 * np.mean([f["minimax-d2"]["score"] for f in fins["dqn-selfplay"]])
    ax.axhline(sp, color=OPP_COLOR["p0"], lw=1.6, ls="--", label="self-play DQN, final")
    ax.set_xlabel("training games (thousands)")
    ax.set_ylabel("score vs minimax-d2 (%)")
    ax.set_ylim(0, 60)
    _style_axes(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fo_curves.pdf")
    plt.close(fig)

    # ---- trainer curves: checkpoints vs own trainer (solid) and the other fixed bots (dashed)
    tp = RES / "fixed_opponent" / "trainer_curves.json"
    if tp.exists():
        T = json.loads(tp.read_text())
        runs = [("vs-random", "random", "trained vs random"), ("vs-heuristic", "heuristic", "trained vs heuristic"),
                ("vs-minimax-d3", "minimax-d3", "trained vs minimax-d3")]
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)
        for ax, (run, trainer, title) in zip(axes, runs):
            for b in ("random", "heuristic", "minimax-d3"):
                by = {}
                for k, v in T.items():
                    if k.startswith(run + "/"):
                        by.setdefault(v["games"], []).append(100 * v[b]["score"])
                g = sorted(by)
                if not g:
                    continue
                own = b == trainer
                ax.plot([x / 1000 for x in g], [np.mean(by[x]) for x in g], color=OPP_COLOR[b],
                        lw=2.4 if own else 1.4, ls="-" if own else "--", marker="o", ms=3.5 if own else 2.5,
                        label=f"vs {b}" + (" (trainer)" if own else ""))
            ax.axhline(50, color=INK, lw=0.8, ls=":")
            ax.set_title(title, fontsize=8.5, color=INK)
            ax.set_xlabel("training games (k)", fontsize=8)
            ax.set_ylim(0, 100)
            ax.set_xticks([50, 100, 150, 200])
            _style_axes(ax)
        axes[0].set_ylabel("score (%)")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=OPP_COLOR[b], lw=1.6, ls="--", marker="o", ms=3, label=f"vs {b}")
                   for b in ("random", "heuristic", "minimax-d3")]
        handles.append(Line2D([], [], color=INK2, lw=2.6, ls="-", label="thick solid = own trainer"))
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(FIG / "fo_trainer_curves.pdf")
        plt.close(fig)

    # ---- round robin between the learner groups
    if not hp.exists():
        return
    H = json.loads(hp.read_text())
    names = [g for g, _, _ in FO_GROUPS]
    n_g = len(names)
    mat = np.full((n_g, n_g), np.nan)
    ci = {}
    rng = np.random.default_rng(0)
    for i, ga in enumerate(names):
        for j, gb in enumerate(names):
            if i == j:
                continue
            M = {}
            for k, v in H.items():
                a, b = k.split("|")
                (an, asd), (bn, bsd) = a.split("/"), b.split("/")
                if an == ga and bn == gb:
                    M[(asd, bsd)] = v["score"]
                elif an == gb and bn == ga:
                    M[(bsd, asd)] = 1 - v["score"]
            if not M:
                continue
            mat[i, j] = 100 * np.mean(list(M.values()))
            S = ["seed0", "seed1", "seed2"]
            boots = [np.mean([M[(a, b)] for a in rng.choice(S, 3) for b in rng.choice(S, 3)]) for _ in range(5000)]
            ci[(ga, gb)] = np.percentile(boots, [2.5, 97.5]) * 100
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100)
    for i in range(n_g):
        for j in range(n_g):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color=INK)
    short = ["vs random", "vs heuristic", "vs minimax-d3", "self-play", "classical\n(Ph. 1)"]
    ax.set_xticks(range(n_g), short, fontsize=7.5, rotation=20)
    ax.set_yticks(range(n_g), short, fontsize=7.5)
    ax.set_xlabel("opponent (trained ...)")
    ax.set_ylabel("player (trained ...)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("player's score (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fo_h2h.pdf")
    plt.close(fig)
    rows = []
    for i, ga in enumerate(names):
        for j, gb in enumerate(names):
            if j > i:
                lo, hi = ci[(ga, gb)]
                la, lb = (FO_GROUPS[i][2].replace("DQN trained ", ""), FO_GROUPS[j][2].replace("DQN trained ", ""))
                rows.append(rf"{la} & {lb} & {mat[i, j]:.1f} & [{lo:.1f}, {hi:.1f}] \\")
    write("fo_h2h.tex", "\n".join(rows) + "\n")


# ------------------------------------------------------------- co-evolution (Phase 4)
CO_CFGS = [("F1-N2", "F1", 2, 200_000), ("F1-N4", "F1", 4, 200_000), ("F1-N8", "F1", 8, 200_000),
           ("F2-N4", "F2", 4, 100_000), ("F2-N8", "F2", 8, 50_000)]


def _co_members(cfg):
    for sd in (0, 1, 2):
        for m in sorted((RES / "coevo" / cfg / f"seed{sd}").glob("m*")):
            yield sd, m


def _co_best(W, cfg, sd, n):
    """Best member of one population by its mean score inside the population (never by test results)."""
    tot = {i: [] for i in range(n)}
    for k, v in W.items():
        run, pair = k.rsplit("/", 1)[0], k.rsplit("/", 1)[1]
        if run != f"{cfg}/seed{sd}":
            continue
        a, b = pair.split("|")
        tot[int(a[1:])].append(v["score"])
        tot[int(b[1:])].append(1 - v["score"])
    return max(tot, key=lambda i: sum(tot[i]) / max(1, len(tot[i])))


def coevo():
    import numpy as np
    ep, wp = RES / "coevo" / "eval_members.json", RES / "coevo" / "within_h2h.json"
    if not (ep.exists() and wp.exists()):
        print("coevo: evaluation missing")
        return
    E, Wn = json.loads(ep.read_text()), json.loads(wp.read_text())
    n_expected = sum(3 * n for _, _, n, _ in CO_CFGS)
    if len(E) < n_expected:
        print(f"coevo: evaluation incomplete ({len(E)}/{n_expected})")
        return
    div = json.loads((RES / "coevo" / "diversity.json").read_text())
    rng = np.random.default_rng(0)

    def avg_vs(entries, pref):
        v = [x["score"] for k, x in entries.items() if k.startswith(pref)]
        return sum(v) / len(v)

    # self-play reference (Phase 3, p = 0)
    L = json.loads((RES / "league" / "dqn_eval.json").read_text())
    sp = {"ppo": np.mean([avg_vs(L[f"p0/seed{s}"], "ppo-s") for s in (0, 1, 2)]),
          "dqn+": np.mean([avg_vs(L[f"p0/seed{s}"], "dqn+-s") for s in (0, 1, 2)]),
          **{b: np.mean([L[f"p0/seed{s}"][b]["score"] for s in (0, 1, 2)]) for b in ("minimax-d2", "minimax-d3", "heuristic", "minimax-d4")}}
    rows, bars = [], {}
    rows.append(rf"self-play DQN (reference) & 1 & 200k & 50 & -- & {pct(sp['ppo'])} & {pct(sp['dqn+'])} & "
                rf"{pct(sp['minimax-d2'])} & {pct(sp['minimax-d3'])} & {pct(sp['heuristic'])} & "
                rf"{pct(sp['minimax-d4'])} & {pct(div['selfplay_seeds']['mean'])} \\")
    rows.append(r"\midrule")
    bars["self-play\n(N=1)"] = {"vs self-play": 50.0, "ppo": 100 * sp["ppo"], "dqn+": 100 * sp["dqn+"],
                                 "minimax-d2": 100 * sp["minimax-d2"], "minimax-d3": 100 * sp["minimax-d3"]}
    for cfg, fam, n, gpm in CO_CFGS:
        # per (population seed, self-play seed): mean over members
        M = {}
        fins, mem_keys = [], []
        for sd, m in _co_members(cfg):
            key = f"{cfg}/seed{sd}/{m.name}"
            mem_keys.append((sd, key))
            fins.append(json.loads((m / "final.json").read_text()))
            for b in (0, 1, 2):
                M.setdefault((sd, b), []).append(E[key][f"selfplay-s{b}"]["score"])
        Mm = {k: np.mean(v) for k, v in M.items()}
        mean = np.mean(list(Mm.values()))
        boots = [np.mean([Mm[(a, b)] for a in rng.choice(3, 3) for b in rng.choice(3, 3)]) for _ in range(10_000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        best = []
        for sd in (0, 1, 2):
            bi = _co_best(Wn, cfg, sd, n)
            best.append(avg_vs(E[f"{cfg}/seed{sd}/m{bi}"], "selfplay-s"))
        ppo = np.mean([avg_vs(E[k], "ppo-s") for _, k in mem_keys])
        dqp = np.mean([avg_vs(E[k], "dqn+-s") for _, k in mem_keys])
        bot = {b: np.mean([f[b]["score"] for f in fins]) for b in ("minimax-d2", "minimax-d3", "heuristic", "minimax-d4")}
        dv = np.mean([div[f"{cfg}/seed{sd}"]["mean"] for sd in (0, 1, 2)])
        rows.append(rf"{fam} & {n} & {gpm // 1000}k & {pct(mean)} [{pct(lo)}, {pct(hi)}] & {pct(np.mean(best))} & "
                    rf"{pct(ppo)} & {pct(dqp)} & {pct(bot['minimax-d2'])} & {pct(bot['minimax-d3'])} & "
                    rf"{pct(bot['heuristic'])} & {pct(bot['minimax-d4'])} & {pct(dv)} \\")
        bars[f"{fam}\nN={n}"] = {"vs self-play": 100 * mean, "vs self-play CI": (100 * lo, 100 * hi), "ppo": 100 * ppo,
                                  "dqn+": 100 * dqp, "minimax-d2": 100 * bot["minimax-d2"], "minimax-d3": 100 * bot["minimax-d3"]}
    write("coevo_results.tex", "\n".join(rows) + "\n")

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    # ---- bars: average member vs self-play and the held-out opponents, per configuration
    keys = [("vs self-play", "vs self-play DQN", OPP_COLOR["p0"]), ("ppo", "vs Phase 2 PPO", OPP_COLOR["ppo"]),
            ("dqn+", "vs Phase 2 DQN+", OPP_COLOR["dqn+"]), ("minimax-d2", "vs minimax-d2", OPP_COLOR["minimax-d2"]),
            ("minimax-d3", "vs minimax-d3", OPP_COLOR["minimax-d3"])]
    groups = list(bars)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(groups))
    w = 0.16
    for j, (k, lab, c) in enumerate(keys):
        vals = [bars[g][k] for g in groups]
        xs = x + (j - 2) * (w + 0.01)
        ax.bar(xs, vals, w, color=c, edgecolor="white", linewidth=0.8, label=lab)
        if k == "vs self-play":
            for xi, g in zip(xs, groups):
                if "vs self-play CI" in bars[g]:
                    lo, hi = bars[g]["vs self-play CI"]
                    ax.errorbar([xi], [bars[g][k]], yerr=[[bars[g][k] - lo], [hi - bars[g][k]]], fmt="none",
                                ecolor=INK2, elinewidth=0.9, capsize=2)
    ax.axhline(50, color=INK, lw=0.8, ls=":")
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7.5)
    ax.set_ylabel("score (%)")
    ax.set_ylim(0, 70)
    _style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "coevo_bars.pdf")
    plt.close(fig)

    # ---- diversity per configuration vs the self-play-seed baseline
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, (cfg, fam, n, _) in enumerate(CO_CFGS):
        v = [100 * div[f"{cfg}/seed{sd}"]["mean"] for sd in (0, 1, 2)]
        ax.scatter([i] * 3, v, color=SERIES1, s=22, zorder=3)
        ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=INK, lw=1.4)
    ax.axhline(100 * div["selfplay_seeds"]["mean"], color=OPP_COLOR["p0"], lw=1.6, ls="--",
               label="independent self-play seeds")
    ax.set_xticks(range(len(CO_CFGS)), [f"{fam} N={n}" for _, fam, n, _ in CO_CFGS], fontsize=7.5)
    ax.set_ylabel("moves that differ (%)")
    ax.set_ylim(40, 70)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "coevo_diversity.pdf")
    plt.close(fig)

    # ---- checkpoint curves vs minimax-d3, average member, x = games per member
    cp = RES / "coevo" / "ckpt_curves.json"
    if cp.exists():
        C = json.loads(cp.read_text())
        fig, ax = plt.subplots(figsize=(6.2, 2.8))
        ramp = {"F1-N2": "#86b6ef", "F1-N4": "#2a78d6", "F1-N8": "#104281", "F2-N4": "#eb6834", "F2-N8": "#a8461d"}
        for cfg, fam, n, _ in CO_CFGS:
            by = {}
            for k, v in C.items():
                if k.startswith(cfg + "/"):
                    # final models sit at the member's actual game count (budget +- a few hundred): snap to the 25k grid
                    g25 = int(round(v["games"] / 25_000) * 25_000)
                    by.setdefault(g25, []).append(100 * v["minimax-d3"]["score"])
            g = sorted(by)
            ax.plot([x / 1000 for x in g], [np.mean(by[x]) for x in g], color=ramp[cfg], lw=2, marker="o", ms=3,
                    ls="-" if fam == "F1" else "--", label=f"{fam} N={n}")
        SC = json.loads((RES / "league" / "dqn_ckpt_curves.json").read_text())
        by = {}
        for k, v in SC.items():
            if k.startswith("p0/"):
                by.setdefault(v["games"], []).append(100 * v["minimax-d3"]["score"])
        g = sorted(by)
        ax.plot([x / 1000 for x in g], [np.mean(by[x]) for x in g], color=OPP_COLOR["p0"], lw=2.2, marker="s", ms=3,
                label="self-play DQN")
        ax.set_xlabel("training games per agent (thousands)")
        ax.set_ylabel("score vs minimax-d3 (%)")
        ax.set_ylim(0, 40)
        _style_axes(ax)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG / "coevo_ckpt.pdf")
        plt.close(fig)

    # ---- non-transitivity inside populations (N >= 3)
    from othello.eval.population_metrics import count_cycles
    rows = []
    for cfg, fam, n, _ in CO_CFGS:
        if n < 3:
            continue
        n_cyc, n_edges, n_pairs = 0, 0, 0
        for sd in (0, 1, 2):
            names = [f"m{i}" for i in range(n)]
            score, games = {}, {}
            for k, v in Wn.items():
                run, pair = k.rsplit("/", 1)[0], k.rsplit("/", 1)[1]
                if run == f"{cfg}/seed{sd}":
                    a, b = pair.split("|")
                    score[(a, b)], games[(a, b)] = v["score"], v["games"]
            cyc, edges = count_cycles(names, score, games)
            n_cyc += len(cyc)
            n_edges += edges
            n_pairs += n * (n - 1) // 2
        rows.append(rf"{fam} & {n} & {n_pairs} & {n_edges} & {n_cyc} \\")
    write("coevo_cycles.tex", "\n".join(rows) + "\n")


def yaml_load(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="results root (default: results/)")
    ap.add_argument("--out", help="report asset root (default: report/)")
    args = ap.parse_args()
    if args.results:
        RES = Path(args.results)
    if args.out:
        TAB, FIG = Path(args.out) / "tables", Path(args.out) / "figures"
    if not args.results:
        throughput(), tuning(), depth(), gate(), diag(), openings(), meta()
    classical()
    selfplay()
    league()
    fixed_opponent()
    coevo()
    print("wrote", sorted(p.name for p in TAB.iterdir()), sorted(p.name for p in FIG.iterdir()))
