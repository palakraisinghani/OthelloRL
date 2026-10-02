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

    def line(label, r, bold_cols=()):
        w = r["weights"]
        cells = [label, f"{w['disc']:g}", f"{w['mobility']:g}", f"{w['corner']:g}", f"{w['xsquare']:g}"]
        cells += [bold_if(pct(r["scores"][o]), o in bold_cols) for o in opp]
        cells.append(bold_if(pct(r["mean_score"]), "mean" in bold_cols))
        return " & ".join(cells) + r" \\"

    shown = [(f"Rank {i}" + (" (selected)" if i == 1 else ""), r) for i, r in enumerate(rows[:5], 1)]
    shown.append(None)
    for k in ["disc", "mobility", "corner", "xsquare"]:
        w = dict(best)
        if w[k] != 0:
            w[k] = 0
            shown.append((f"Selected, no {k.replace('xsquare', 'X-square')}", find(w)))
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
        label, r = x
        bc = [o for o in opp if r["scores"][o] == col_best[o]] + (["mean"] if r["mean_score"] == col_best["mean"] else [])
        out.append(line(label, r, bc))
    write("tuning.tex", "\n".join(out) + "\n")
    write("tuning_meta.tex", rf"\newcommand{{\NumCandidates}}{{{len(rows)}}}" + "\n")


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
    out, elo = [], []
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
        e = v["elo"]
        elo.append(rf"{n}$\times${n} & {e['random']:.0f} & {e['heuristic']:.0f} & {e['minimax-d4']:.0f} \\")
    write("gate.tex", "\n".join(out[:-1]) + "\n")
    write("elo.tex", "\n".join(elo) + "\n")


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
        hours, _ = _ms([r["final"]["train_seconds"] / 3600 for r in have[k]])
        cells += [pct(wr), f"{hours:.1f}"]
        rows.append(" & ".join(cells) + r" \\")
    # reference row: the heuristic bot itself, from the rule-based round robin
    g6 = {(m["a"], m["b"]): m for m in load("baselines/gate.json")["sizes"]["6"]["matches"]}
    h_r = g6[("heuristic", "random")]
    h_m = g6[("minimax-d4", "heuristic")]
    rows.append(r"\midrule")
    rows.append(rf"\textit{{Heuristic bot (reference)}} & -- & {pct(h_r['score'])} & -- & "
                rf"{pct(1 - h_m['score'])} & -- & -- \\")
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
    print("wrote", sorted(p.name for p in TAB.iterdir()), sorted(p.name for p in FIG.iterdir()))
