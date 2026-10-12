# Othello RL: Step-wise Plan

The steps implement `othello_rl_project_roadmap.md`. Do them in order. Each step has a **Done when** check, and you only move on once it passes.

**Default stack:** Python 3.11, NumPy for the env, PyTorch for the networks, pytest, matplotlib. Each run uses one YAML config, writes its logs to `results/` as CSV/JSONL, and has its seeds fixed.

---

## Phase 0: Environment & Baselines (~1 week) — ✅ done

**Step 0.1 — Project skeleton**
Create `othello/` (`env.py`, `agents/`, `eval/`, `rl/`, `utils/`) plus `configs/`, `scripts/`, `tests/` and `results/`. Add `requirements.txt` and a seeding/config/logging helper in `othello/utils/`.
*Done when:* `pytest` runs (with zero tests) and `python -c "import othello"` works.

**Step 0.2 — Board representation**
`OthelloEnv(board_size)` with the standard 4-disc start position, the side to move, and a cheap `clone()`. The size must never be hardcoded.
*Done when:* the initial board for sizes 6, 8 and 10 prints correctly.

**Step 0.3 — Legal moves + flipping**
Scan the 8 directions, flip the captured discs, and return the action mask with length `n*n + 1`, where the last index is "pass".
*Done when:* hand-checked test positions pass, and the perft counts from the 8x8 start match the known values (depth 1–4: 4, 12, 56, 244).

**Step 0.4 — Pass + terminal + scoring**
A pass is forced when the player has no legal move. The game ends after two consecutive passes or a full board. Report the winner and disc difference.
*Done when:* tests cover a forced pass, a double-pass end, and a full-board end.

**Step 0.5 — State encoding**
`encode(state, planes=[...])` with configurable planes: mine, theirs, empty, turn, legal mask.
*Done when:* the output shape is `(C, n, n)` for each plane config and each size.

**Step 0.6 — Agent interface + Random bot**
`Agent.act(state) -> action` in `agents/base.py`; `agents/random.py`.

**Step 0.7 — Heuristic bot**
Score = disc count + mobility + corner occupancy, using position weights generated from the board geometry (corners, X/C-squares, edges). Use random tie-breaking or a small ε so the bot isn't fully deterministic, which stops tabular methods from memorizing a single line against it.

**Step 0.8 — Alpha-beta minimax bot**
Depth 4–6, uses the heuristic as its evaluation, with move ordering.

**Step 0.9 — Eval harness**
`eval/arena.py`: plays N games between any two agents with colors swapped, starting from K randomized opening positions, using multiprocessing. Reports W/D/L with confidence intervals. `eval/replay.py` saves and prints games. (`eval/elo.py`, a Bradley–Terry Elo fitter, exists but is only used in the final step F.0.)

**Step 0.10 — Tune heuristic weights**
Run a short grid or random search over the weights, playing against random and against earlier weight settings.

**Step 0.11 — Sanity gate**
*Done when:* on 6x6 over 1,000 games, heuristic beats random ≥95% and minimax beats heuristic ≥80%. Record env throughput (games/sec); it is needed for compute budgeting later.

---

## Phase 1: Traditional Methods on 6x6 (~1.5 weeks) — ✅ done

**Step 1.1 — Shared training/eval loop**
`scripts/train.py --config ...` trains against the heuristic bot and evaluates every N episodes on fixed eval seeds, writing the curve to CSV. All methods get the same training-game budget and 3 seeds each.

**Step 1.2 — Tabular Q-learning**
`rl/tabular.py` with a dict table keyed by the canonical state (smallest form under the 8 symmetries). Also log table coverage: the number of states and the fraction revisited.

**Step 1.3 — SARSA**
Same file, using the on-policy update rule.

**Step 1.4 — First-visit MC control**
Same file. Also log the variance of the returns.

**Step 1.5 — Linear FA Q-learning**
`rl/linear.py` with features for disc diff, mobility, corners and edge stability. A `use_disc_features` flag supports the ablation.

**Step 1.6 — Disc-count ablation**
Run linear FA with disc features heavily weighted vs. without them.

**Step 1.7 — Vanilla DQN**
`rl/dqn.py` with an MLP or small CNN, replay buffer, target net and illegal-move masking, trained against the heuristic bot.

**Step 1.8 — Phase 1 plots + writeup**
`scripts/plot_phase1.py` draws all 5 curves on one chart with horizontal lines for random, heuristic and minimax, and also produces the ablation plot. Write a "why each fails" section that cites the logged coverage and variance numbers.
*Done when:* both plots and the writeup section exist.

---

## Phase 2: Modern Self-Play Methods on 6x6 (~2.5 weeks) — ✅ done (winner: AlphaZero seed 1 = Oroletto)

**Step 2.1 — Opponent pool**
A shared module that saves checkpoint snapshots and samples opponents from them. It is used by both DQN+ and PPO.

**Step 2.2 — Self-play DQN+**
Extend `dqn.py` with Double DQN and opponent-pool self-play.

**Step 2.3 — Self-play PPO**
`rl/ppo.py`: shared actor-critic, masked softmax, GAE, opponent pool.

**Step 2.4 — AlphaZero: network**
`rl/alphazero/net.py`: small ResNet with policy and value heads.

**Step 2.5 — AlphaZero: MCTS**
`mcts.py`: PUCT, Dirichlet noise at the root, temperature schedule, legal-move masking. Batch the network calls from the start, because later phases depend on MCTS being fast.
*Done when:* MCTS with a dummy uniform network beats random.

**Step 2.6 — AlphaZero: self-play + training loop**
`selfplay.py` (parallel workers) and `train.py` (replay window, periodic eval/gating), plus `symmetry.py` for 8-fold dihedral augmentation.

**Step 2.7 — Run all 3 methods**
3 seeds each, evaluated against heuristic and minimax.

**Step 2.8 — Comparison table**
Columns: final win rate, games to reach the threshold (e.g. 80% vs heuristic), wall-clock time, and variance across seeds. As an extra check, 6x6 is weakly solved (perfect play ends 16–20, a second-player win), so compare the best agent's self-play against that result.
*Done when:* the table and checkpoints are saved and a winner is chosen. The winner is released as **Oroletto** (may be replaced by a stronger Phase 4 model): point `models/oroletto.json` at its run directory so `othello --play oroletto` loads it.

**Step 2.9 — Motivation: the frozen target of a fixed opponent** — ✅ done (frozen target clear for random/heuristic; self-play beats every fixed-opponent DQN head-to-head)
- **Why:** to motivate self-play. A fixed opponent is a *frozen learning target*. Once the learner beats it reliably, almost every game is a win, which carries little information, so progress stalls. Self-play has no frozen target: the opponent improves with the learner. ("Capped at the opponent's strength" is too strong a claim: Phase 1's DQN, trained only against heuristic, beats it 65% of the time.)
- **Objective & expectation:**
  - **Objective:** show how training against a fixed opponent stalls, compared with self-play.
  - **Expectations** (hypotheses):
    - **(a)** each learner's score *against its own trainer* rises and then plateaus, while its general strength (against the other bots and the other learners) stops improving;
    - **(b)** strength across learners orders by trainer (random < heuristic < minimax-d3), mainly because a stronger trainer gives a better learning signal, not a higher "ceiling";
    - **(c)** the self-play DQN (Phase 3, p = 0) matches or beats every fixed-opponent DQN.
  - **Possible inverted U:** against random almost every game is a win, and against minimax-d3 most are losses. Both ends are weak signals, so the d3 learner may not end up strongest. That would still favour self-play, which is always evenly matched.
- **Methodology:**
  - Three learners (DQN1, DQN2, DQN3), identical to Phase 3's self-play DQN (Double DQN, same network, buffer and schedule). The only difference is the training opponent: random, heuristic (ε = 0.05) or minimax-d3.
  - **Comparison groups:**
    - the three fixed-opponent learners;
    - the self-play DQN;
    - **Phase 1's classical (vanilla) DQN**, also trained only against heuristic, which also shows whether Double DQN matters.
- **Experiment details:**
  - **Config:** `configs/fixed_opponent_dqn.yaml`, with runs `vs-random`, `vs-heuristic`, `vs-minimax-d3` × seeds {0, 1, 2} × 200k games; random 6-ply openings; alternating colours; checkpoints every 50k games.
  - **Learning curve:** every 5k games, 200 games against **minimax-d2**, which none of the three trains against.
  - **Final evaluation** (openings seed 4243): random, heuristic and d2/d3/d4 (1,000 games each). The Phase 1 classical DQN is additionally evaluated against d2 and d3 (stage `panel`).
  - **Trainer curves (central figure):** each run's checkpoints at 50k, 100k, 150k and 200k play its own trainer and each other fixed bot, 200 games each (stage `trainer`).
  - **Head-to-head round robin:** 5 groups (DQN-random, DQN-heuristic, DQN-d3, self-play DQN, classical DQN), all 9 seed pairs per group pair, 200 games each. That's 10 group pairs × 1,800 games (stage `h2h`).
  - **Statistics:** mean ± std over seeds; 95% CIs per pairing; a seed-level bootstrap CI for each group-vs-group score. No pass/fail criteria are fixed in advance.
  - **Output:** `results/fixed_opponent/`. The report gets a new first subsection of Section 5, "Why self-play? The frozen target of a fixed opponent", which answers the boxed *Point to Ponder* in Section 4.3.4.
  - **Resources:** 9 runs (the d3 arm is ~2.5× slower) plus ~1 h of evaluation.
- *Done when:* the trainer curves, the bot-panel table, the bars and the 5×5 head-to-head matrix are in the report, with hypotheses (a)–(c) discussed.

---

## Phase 3: Does Opponent Variety Help DQN? (6x6, ~1 week) — ✅ done (no mix beats self-play; strength falls with p)

**Question:** does training DQN against a *variety* of strategies beat training it by pure self-play?
- **Learner:** DQN only (Double DQN, the Phase 2 DQN+ settings). PPO and AlphaZero are not part of this phase.
- **League:** weak agents only: the 3 rule-based bots, the 5 traditional agents, and the learner's own past checkpoints.
- **Sampling:** opponents are drawn uniformly. AlphaStar-style prioritised sampling (PFSP) may be added later, so the sampler is pluggable and per-opponent results are always logged.
- **Mixing variable:** the league share **p**, the fraction of training games played against a league member; the rest are against the live learner.

**Hypotheses going in** (predictions to test, not results):
- **H1, robustness:** mixing in varied opponents makes DQN more robust. It should lose less to unusual play and score higher against weak bots.
- **H2, signal:** a league of *weak* agents gives a poor training signal once they are beaten, because almost every game is then a win and carries little information. So league-only training (p = 1) should end up weaker than self-play on held-out tests.
- **H3, best mix:** an intermediate mix keeps self-play's matched-strength curriculum while adding variety. If anything beats self-play, it should be a mix, and held-out strength should peak at an intermediate p.
- **H4, replay dilution:** DQN learns off-policy from a replay buffer. At high p, the buffer fills with easy wins against weak opponents, whose bootstrapped values are uninformative about strong play, which would slow learning. This predicts the drop at high p appears early in training and grows over time.

---

**Step 3.1 — Save/load the traditional agents** — ✅ done
- **Why:** the league needs the 5 traditional agents as playable opponents. The linear and DQN models were saved in Phase 1, but the Q-learning, SARSA and MC tables were not. They only ever existed in memory during training.
- **Objective & expectation:**
  - **Objective:** all 8 league agents load from disk and play through the normal agent interface.
  - **Expectation:** Phase 1 seeding was fully deterministic (fixed random seeds, integer-keyed dicts), so re-training seed 0 should reproduce *exactly* the same tables and the same `final.json` numbers. If it doesn't, the re-trained versions are used and the difference is reported.
- **Methodology:**
  - Add `save`/`load` to `TabularLearner` (a gzipped pickle of the flat Q dict, plus visit counts for MC), and a tabular branch in `othello/models.py:load_model`.
  - Re-run `scripts/train_rl.py --run {q-learning,sarsa,mc-control} --seed 0` in parallel (~30 min). Save `table.pkl.gz`, about 100 MB each, which is git-ignored.
  - Compare the new `final.json` against the committed Phase 1 files.
- **Experiment details:**
  - **Runs:** `python scripts/train_rl.py --run {q-learning|sarsa|mc-control} --seed 0`, all three in parallel. The config is unchanged from Phase 1 (`configs/classical.yaml`): 200k games against heuristic (ε = 0.05), α = 0.1 (MC: 1/N), ε-greedy 0.3 → 0.05 over the first 100k games.
  - **Output:** `results/classical/<run>/seed0/table.pkl.gz` (~100 MB each; git-ignored), plus a regenerated `final.json` and `curve.csv`.
  - **Check:** `diff` the new `final.json` against `git show HEAD:results/classical/<run>/seed0/final.json`. Every score must be identical to the last digit.
  - **Load test:** load each saved agent through `load_model`, play 20 games against random, and confirm it never makes an illegal move and gets the same score as the live learner over the same openings.
  - **Resources:** ~30 min wall-clock, ~2 GB RAM in total.
- *Done when:* each re-trained seed-0 `final.json` equals the Phase 1 one, a save → load round-trip test passes, and all 8 agents load through `load_model` / `make_agent`.

**Step 3.2 — League + sampler** — ✅ done
- **Why:** this is the core mechanism of the phase. It controls *who* DQN plays and records *how it fares against each opponent*. The second part is needed both for the analysis and for any later switch to PFSP, which samples by win rate.
- **Objective & expectation:**
  - **Objective:** one component that, for every training game, returns either the live learner (self-play) or a league member.
  - **Expectation:** observed sampling frequencies match p within sampling noise. In training logs, DQN should overtake the weakest members (random, tabular agents) almost immediately and minimax-d4 last, if at all, since DQN only scores ~19% against d4.
- **Methodology:**
  - Extend `OpponentPool` (`othello/rl/pool.py`) into a `League`.
    - **Fixed members** are built once from specs with `make_agent` / `load_model`.
    - **Snapshots:** a frozen copy of the learner is added every K = 5,000 games (~40 per 200k-game run); at most 20 are kept, oldest dropped first.
  - Each game: with probability p, pick a league member (two-level: fixed agents vs snapshots, 50/50, then uniform within the group); otherwise the live learner.
  - The sampler is a small strategy object (`UniformSampler` now; `PFSPSampler` later can reuse the same per-opponent W/D/L counts).
  - `stats()` writes per-opponent W/D/L for the last evaluation window into the curve CSV.
  - No new training loop is needed: `othello/rl/common.py:train` already takes a pool and draws an opponent per game, and `DQNLearner` already provides `snapshot()`.
- **Experiment details:**
  - **League members (fixed, 8):**
    - `random`;
    - `heuristic` (tuned weights, ε = 0.05);
    - `minimax-d4` (tuned weights);
    - Phase 1 seed-0 `q-learning`, `sarsa`, `mc-control`, `linear` and classical `dqn`, all playing greedily.
  - **Snapshots:** frozen copies of the learner, added at games 5k, 10k, …; at most 20 kept, oldest first out; ε = 0.05 when playing.
  - **Sampling:** each game picks the live learner with probability 1 − p, and a league member with probability p. **Within the league, sampling is two-level:** 50% one of the 8 fixed agents and 50% one of the snapshots, uniform within each group. This keeps the league's make-up constant over training; plain uniform sampling would let snapshots grow to 20 of 28 members (71%). Until the first snapshot exists, all league games go to fixed agents.
  - **Logging:** per-opponent W/D/L and the realised mix (self vs fixed vs snapshot) for every 5k-game window, written to `curve.csv`.
  - **Tests:** over 10,000 sampled games with p ∈ {0, 0.5, 1}, the observed shares are within ±2 percentage points of expected.
  - **Smoke run:** p = 0.5, 10k games, one seed, evaluated every 2.5k games (~5 min).
- *Done when:* unit tests confirm the frequencies for p = 0, 0.5 and 1, and a smoke run's CSV contains per-opponent columns.

**Step 3.3 — Held-out evaluation panel** — ✅ done (DQN+ scores 33% vs d2, 25% vs d3: accepted)
- **Why:** heuristic and minimax-d4 are now *training* opponents, so beating them proves less. A learner could memorise them, as MC control memorised the heuristic bot in Phase 1. "Better than self-play" must be judged against opponents DQN never trains on. They must also be close to DQN's strength (~19% against d4), or every mix would score near 0% or 100% and the differences would vanish.
- **Objective & expectation:**
  - **Objective:** a held-out panel that spans DQN's strength range and can separate the mixes.
  - **Expectation:** Phase 0's depth sweep places minimax-d2 and d3 between the heuristic bot and d4, so DQN should score roughly 30–60% against them: the most informative range.
- **Methodology:** the held-out panel is:
  - **minimax-d2 and minimax-d3:** not in the league. They share the heuristic's evaluation function, so they are related to two league members (heuristic and d4). This is partial overlap, which the report states.
  - **Phase 2 PPO and Phase 2 DQN+:** completely unseen learned strategies of similar strength.
  - **Head-to-head against the p = 0 model** at the same budget.

  Calibrate first: Phase 2 DQN+ against d2 and d3 (400 games each), to confirm the panel is neither too easy nor too hard.
- **Experiment details:**
  - **Panel:** `minimax-d2` and `minimax-d3`, both using the tuned heuristic weights as their evaluation, plus the three Phase 2 PPO seeds (`results/selfplay/ppo/seed{0,1,2}`) and the three Phase 2 DQN+ seeds (`results/selfplay/dqn-plus/seed{0,1,2}`).
  - **Calibration matches:** each Phase 2 DQN+ seed plays d2 and d3 for 200 openings × both colours = 400 games, giving 1,200 games per depth in total. Openings are 6 random plies, with seed 5151, which is not used anywhere else.
  - **Calibration rule:** the panel is accepted if DQN+'s mean score against each of d2 and d3 lies in 20–80%. If d3 is above 80%, add d4-with-the-hand-default-evaluation (not a league member) as a harder held-out depth. If d2 is below 20%, drop it.
  - **Reference strengths already known** from Phase 0 (against the heuristic bot): d2 73.5%, d3 86.1%, d4 88.2%.
  - **Output:** `results/league/panel_calibration.json`; runtime ~10 min on 8 cores.
- *Done when:* DQN+ scores between ~20% and 80% against d2 and d3. Otherwise adjust the depths.

**Step 3.4 — DQN runs across mixes** — ✅ done (15 runs, 1.8–2.2 h each)
- **Why:** to measure the effect of mixing as a dose-response curve, i.e. how strength changes as the league share goes from 0 to 1, rather than at two or three points.
- **Objective & expectation:**
  - **Objective:** DQN strength and robustness at p ∈ {0, 0.25, 0.5, 0.75, 1}.
  - **Expectation:**
    - In-league scores (heuristic, d4) should rise with p, because DQN trains directly on them.
    - Held-out scores should peak at an intermediate p and be lowest at p = 1 (H2, H3).
    - Per H4, the learning curves of high-p runs should fall behind early.
- **Methodology:**
  - `configs/league_dqn.yaml` with runs `p0`, `p25`, `p50`, `p75` and `p100`, each × seeds {0, 1, 2} × 200k games.
  - The learner is identical to Phase 2 DQN+: Double DQN, the same network, replay buffer and ε schedule. It learns only from its own moves.
  - The league contains Phase 1's *classical* DQN as a fixed member; that is a separate, weak agent, not the learner.
  - p = 0 is *re-run* under the new code (pure live self-play), so all mixes share one implementation. Phase 2 DQN+ trained partly against a pool of its own past snapshots, a slightly different baseline, and is kept as a reference and as a held-out opponent.
  - 15 runs on 8 cores (~1.3 h each), with output to `results/league/dqn/<mix>/seed<k>/`.
  - Checkpoints are saved every 25k games, for the learning curves against the held-out panel. The in-training learning curve is measured against minimax-d3 (held-out), not heuristic (now in-league).
- **Experiment details:**
  - **Config:** `configs/league_dqn.yaml`, with runs `p0`, `p25`, `p50`, `p75`, `p100` × seeds {0, 1, 2} = 15 runs.
  - **Learner (identical to Phase 2 DQN+):**
    - MLP 108 → 256 → 256 → 37 on planes (mine, theirs, legal);
    - Double DQN; Adam, learning rate 5×10⁻⁴; Huber loss;
    - replay buffer 200k; batch 128; one update every 4 learner moves; target sync every 2,000 updates; warm-up 5,000 transitions;
    - γ = 1; ε-greedy 0.3 → 0.05 over the first 100k games; 200k games per run.
  - **Training games:** start from 6-ply random openings, drawn from a 20k-opening pool per seed; colours alternate every game.
  - **Learning curve:** every 5k games, the greedy policy plays 100 openings × 2 = 200 games against **minimax-d3**, a held-out opponent, so the curve is not inflated by memorisation. Phases 1–2 used heuristic, which is now in the league. Evaluation openings come from seed 4242.
  - **Checkpoints:** `ckpt/g{games}.pt` every 25k games (8 per run), for the H4 analysis in 3.5.
  - **Logged per 5k window:** the realised opponent mix, per-opponent W/D/L, mean loss, number of updates, and the share of replay-buffer transitions that came from league games (for H4).
  - **Resources:** 15 runs on 8 workers; ~1.3 h each without league games, possibly up to ~2 h at high p because minimax-d4 is slow in pure Python. Expected total ~3–5 h.
- *Done when:* all 15 runs have `final.json`, curves with per-opponent columns, and checkpoints (~3–4 h).

**Step 3.5 — Held-out evaluation** — ✅ done
- **Why:** this step decides the phase's question, using only opponents DQN never trained on.
- **Objective & expectation:**
  - **Objective:** a fair, statistically grounded answer to "does mixing improve DQN, and at which p?"
  - **Expectation:** per H1–H3, held-out strength is an inverted U in p, while in-league strength rises monotonically. Comparing the two curves exposes any memorisation.
- **Methodology:** `scripts/eval_phase3.py`. For every model (5 mixes × 3 seeds):
  1. Head-to-head against the p = 0 models at the same budget: all seed pairs, 100 openings × both colours each.
  2. Against minimax-d2 and d3.
  3. Against Phase 2 PPO and Phase 2 DQN+.
  4. Against the in-league bots, reported but flagged "in-league".

  Checkpoint learning curves against the held-out panel test H4. Uncertainty comes from 95% CIs per pairing plus the spread across seeds. A difference is called real only if it holds across seeds and its CI excludes 50%.
- **Experiment details:**
  - **Models:** 15 (5 mixes × 3 seeds), each at its final 200k-game checkpoint, playing greedily with no exploration.
  - **Matches per model** (openings from seed 4243, 6 plies, both colours):

    | Opponent | Games | Held-out? |
    |---|---|---|
    | each of the 3 p = 0 seeds (skipped for the model itself) | 200 each = 600 | yes |
    | minimax-d2, minimax-d3 | 1,000 each | yes (partial overlap noted) |
    | each Phase 2 PPO seed, each Phase 2 DQN+ seed | 200 each = 1,200 | yes |
    | random, heuristic, minimax-d4 | 1,000 each | **no**, in-league |

  - **Head-to-head between mixes:** every mix plays every other mix over all 9 seed pairs × 200 games, for the full matrix.
  - **Checkpoint curves (H4):** each run's 8 checkpoints play d3 and Phase 2 PPO seed 0 (200 games each).
  - **Statistics:**
    - per pairing: score with a 95% normal-approximation CI;
    - per mix: mean ± std over seeds;
    - mix vs p = 0: the mean over all 9 seed pairs, with a seed-level bootstrap CI (resampling seeds of both sides, 10,000 resamples).
    - **Decision rule:** a mix "beats self-play" only if its bootstrap CI against p = 0 lies entirely above 50%, *and* it is not worse than p = 0 against any held-out panel opponent.
  - **Output:** `results/league/dqn_eval.json`, `results/league/dqn_h2h_matrix.json` and `results/league/dqn_ckpt_curves.json`.
  - **Resources:** ~0.25 M games; cheap MLP agents and shallow minimax, so ~1–2 h on 8 cores.
- *Done when:* `results/league/dqn_eval.json` exists with all pairings, plus checkpoint curves.

**Step 3.6 — Report section "Does Opponent Variety Help DQN?"** — ✅ done
- **Why:** documents the result and its interpretation, including the possibility that self-play wins.
- **Objective & expectation:** a section in the same style as the earlier ones. Its central figure plots score against p, with held-out and in-league curves side by side, so any memorisation is visible at a glance.
- **Methodology:**
  - Assets from `scripts/make_report_assets.py`:
    - a league-membership table;
    - the score-vs-p figure (held-out vs in-league);
    - a bar graph of the best mix vs p = 0 against every panel opponent;
    - per-opponent training win-rate curves;
    - a head-to-head matrix between mixes;
    - checkpoint learning curves.
  - Prose follows the experiment-report skill: trend → mechanism → component role → implication, with hypotheses labelled, and H1–H4 each confirmed or refuted explicitly.
- **Experiment details:**
  - **Figures** (`report/figures/`):
    - `league_mix_curve.pdf`: score against p for held-out vs in-league opponents, mean over seeds with min–max whiskers;
    - `league_best_vs_p0.pdf`: grouped bars for the best mix vs p = 0 against every panel opponent;
    - `league_opponent_curves.pdf`: training win rate against each league member over time;
    - `league_h2h.pdf`: the 5 × 5 head-to-head heat-map between mixes (diverging palette, gray at 50%);
    - `league_ckpt_curves.pdf`: held-out score against training games, per mix (H4).
  - **Tables** (`report/tables/`):
    - `league_members.tex`: members, type and strength (score against the heuristic bot from earlier phases);
    - `league_results.tex`: per mix, held-out and in-league scores with CIs;
    - `league_h4.tex`: the share of buffer data from league games vs held-out learning speed.
  - **Generation:** `league()` in `scripts/make_report_assets.py`, run as `python scripts/make_report_assets.py && tectonic report.tex`.
  - **Placement:** the section goes after "Self-Play Methods"; Next Steps is updated.
- *Done when:* the report compiles with no undefined references and the new pages have been checked visually.

---

## Phase 4: Multi-Agent Co-Evolution vs Self-Play (6x6, ~1.5 weeks) — ✅ done (co-evolution ties self-play at equal games per agent; weaker at equal total games; no extra diversity, no cycles)

**Question:** do several *independent* DQN agents (separate networks and learners), learning only by playing each other, become better or more robust than DQN trained by self-play?

**Design (decided 2026-10-05):**
- **Learner:** DQN first (identical to the Phase 3 / Step 2.9 self-play DQN). An AlphaZero population is optional, run only if DQN shows an effect.
- **Population sizes:** N ∈ {2, 4, 8}.
- **Pairing:** uniform. Each game is between two different members chosen at random; members never play their own copy. Strength-matched and PFSP pairing are deferred.
- **Budget, reported both ways:**
  - **F1 (primary): equal games per agent.** Each member plays 200k games, like each self-play DQN.
  - **F2: equal total games.** The population shares 200k games, so 2 × 200k / N per member. N = 2 is identical in F1 and F2.

**Motivation:**
- Phase 3 found that a *weak, fixed* league hurts DQN. But it changed variety and opponent strength at once, since most league games were lopsided. In co-evolution the opponents are varied *and* adaptive, and roughly matched because they learn too. That separates "variety" from "mismatched strength".
- Self-play sees only its own style. Independent agents may find different strategies and probe each other's weaknesses (competitive co-evolution; population-based training; AlphaStar's league).
- Populations can cycle (rock-paper-scissors) instead of improving. Whether Othello allows this is itself a result.
- Each co-evolution game trains *two* learners, whereas self-play DQN learns from one side only.

**Hypotheses** (predictions to test):
- **H1, per-agent strength (F1):** the average member is about level with self-play DQN.
- **H2, matched variety:** co-evolution does **not** show Phase 3's decline. If it does, variety itself hurts DQN, not just mismatched opponents.
- **H3, diversity:** members' policies differ *more* than independent self-play seeds do. If not, the population has collapsed into one strategy.
- **H4, robustness:** any gain is largest against unseen *learned* opponents (Phase 2 PPO and DQN+) rather than the minimax bots.
- **H5, transitivity:** Othello is largely transitive, so there are few or no significant rock-paper-scissors cycles among members.
- **H6, population size:** larger N gives more diversity but slower early progress per agent.
- **H7, equal compute (F2):** at N = 4, training two learners per game partly offsets the split budget. At N = 8 (only 50k games each), members are weaker than self-play.

---

**Run order:** Phase 4's training (Steps 4.3–4.4, all 15 runs via `scripts/train_coevo.py --all`) starts automatically once Step 2.9's evaluation has finished, so the two don't compete for cores.

**Step 4.1 — Population trainer (DQN)** — ✅ done (`othello/rl/population.py`, `scripts/train_coevo.py`, `configs/coevo_dqn.yaml`; smoke run OK)
- **Why:** the existing training loop (`othello/rl/common.py:train`) trains one learner against an opponent. Co-evolution needs N learners trained in the same games.
- **Objective & expectation:** a trainer in which each game pairs two members and **both** learn from their own moves. Each member's game count, exploration schedule and logging behave exactly as for a single DQN, so results are directly comparable.
- **Methodology:**
  - New `othello/rl/population.py`: a list of N `DQNLearner`s, each with its own seed, network, replay buffer and optimiser.
  - **Each game:**
    1. draw an ordered pair (i, j), i ≠ j, uniformly;
    2. alternate colours;
    3. start from a random 6-ply opening;
    4. both members act ε-greedily under their own ε schedule, which runs on their own game count;
    5. after the game, `learner_i.end_episode(traj_i)` and `learner_j.end_episode(traj_j)`.
  - Reuses the evaluation, checkpointing and CSV logging of `common.train`.
- **Experiment details:**
  - **Logging:** per member, every 5k of *its own* games: score against minimax-d3 (held-out, 200 games) and score against each other member in training, giving the win-rate matrix over time.
  - **Checkpoints:** every 25k of each member's games.
  - **Config:** `configs/coevo_dqn.yaml`, with keys `population: N`, `games_per_member`, `pairing: uniform`.
  - **Tests:**
    - pairs are uniform and never i = i;
    - every member's game count is within ±2% of the others after 10k games;
    - both trajectories are recorded with the right player perspective.
  - **Smoke run:** N = 2 for 5k games per member.
- *Done when:* tests pass and the smoke run logs both members' curves and their pairwise scores.

**Step 4.2 — Diversity and non-transitivity measures** — ✅ done (baseline: independent self-play DQN seeds disagree on 57% of positions)
- **Why:** H3 and H5 need measures that strength scores alone don't give.
- **Objective & expectation:** two reusable metrics, each with a baseline from *independent self-play seeds*, because some diversity exists even without co-evolution.
- **Methodology:**
  - **Policy disagreement:** a fixed set of 1,000 positions, sampled from games between the reference bots and earlier learners (seed 6161). For two agents, the fraction of positions where their greedy moves differ.
    - Population diversity = mean over member pairs.
    - Baseline = mean over pairs of the 3 Phase 3 self-play DQN seeds.
  - **Non-transitivity:** from the head-to-head matrix between members (final, and per checkpoint), count directed 3-cycles A > B > C > A where every edge's 95% CI excludes 50%.
- **Experiment details:** `scripts/population_metrics.py`, with output to `results/coevo/diversity.json` and `results/coevo/cycles.json`. Unit tests on toy matrices with a known cycle and on a transitive matrix.
- *Done when:* the metrics run on the Phase 3 self-play seeds and give the baseline disagreement.

**Step 4.3 — F1 runs: equal games per agent** — ✅ done
- **Why:** the primary, scientific comparison: the same experience per agent, against varied adaptive opponents instead of itself.
- **Objective & expectation:** N ∈ {2, 4, 8} × 3 seeds. Members roughly level with self-play (H1), with no Phase 3-style decline (H2).
- **Methodology:** 200k games per member, so 200k / 400k / 800k total games per run. Hyperparameters are identical to the self-play DQN.
- **Experiment details:**
  - 9 runs in parallel on 8 cores.
  - Estimated wall time is dominated by N = 8: ~800k games, with 2 learners updating per game, ~10 h. N = 2 and 4 take ~2.5 and ~5 h.
  - Output to `results/coevo/F1/N{n}/seed{k}/m{i}/`.
- *Done when:* all members of all runs have `final.json` and checkpoints.

**Step 4.4 — F2 runs: equal total games** — ✅ done
- **Why:** the practical comparison: with one self-play run's budget, is a population a better use of it?
- **Objective & expectation:** N ∈ {4, 8} × 3 seeds with 200k total games: 100k / 50k per member. N = 2 is the F1 run. Members weaker at N = 8. N = 4 is the interesting case (H7).
- **Methodology:** same trainer; `games_per_member = 2 × 200k / N`. The ε schedule decays over the first half of *each member's* budget, as everywhere else.
- **Experiment details:** 6 runs, each about the cost of one self-play run (~2–3 h); they run in parallel after or alongside F1 if cores allow. Output to `results/coevo/F2/N{n}/seed{k}/`.
- *Done when:* all runs have `final.json` and checkpoints.

**Step 4.5 — Evaluation against self-play and the held-out panel** — ✅ done (minimax scores taken from each member's end-of-training evaluation: 500 games per depth)
- **Why:** decides H1, H2, H4, H6 and H7, with the same protocol as Phase 3 and Step 2.9, so all of them are comparable.
- **Objective & expectation:** per configuration, the average member and the best member compared with the self-play DQN (Phase 3, p = 0, 3 seeds).
- **Methodology:** for every member of every run:
  - head-to-head against the 3 self-play DQN seeds (200 games each);
  - minimax-d2 and d3 (1,000 games each);
  - Phase 2 PPO and DQN+ (3 seeds each, 200 games each);
  - random, heuristic and minimax-d4 (1,000 games each).
- **Experiment details:**
  - **Primary statistic:** the **average member's** score against self-play, with a seed-level bootstrap 95% CI (resampling population seeds and self-play seeds).
  - **Best member:** chosen using **games within the population only**, never the test results, and reported flagged as best-of-N.
  - **Checkpoint curves:** each member's checkpoints against minimax-d3 and PPO seed 0 (H6: early-progress speed).
  - Output: `results/coevo/eval.json` and `results/coevo/ckpt_curves.json`; script `scripts/eval_coevo.py`, modelled on `scripts/eval_phase3.py`.
- *Done when:* every model's results file exists, with bootstrap CIs and checkpoint curves.

**Step 4.6 — Population dynamics** — ✅ done
- **Why:** does co-evolution produce progress, cycling or collapse (H3, H5)?
- **Objective & expectation:** strength over time per member, diversity against the self-play baseline, and cycle counts. Expected: diversity above baseline, few or no significant cycles.
- **Methodology:** apply the Step 4.2 metrics to every population:
  - final members: disagreement and the head-to-head matrix (all member pairs, 200 games each);
  - each member's checkpoints: the win-rate matrix over time, from the training logs and checkpoint matches.
- *Done when:* figures exist for strength over time, diversity vs baseline, and cycle counts.

**Step 4.7 (optional) — AlphaZero population** — ⏭ skipped (DQN showed no co-evolution effect; the batched runner was never needed: Phase 6 was cancelled)
- **Why:** run only if DQN shows a co-evolution effect worth confirming with the strongest learner.
- **Methodology:**
  - First build the **generic batched game runner** (`run_games` in `othello/rl/alphazero/selfplay.py`). Each side is an MCTS player or a plain `Agent`, with search batched per MCTS player. `az_evaluate` and self-play become thin wrappers, and the existing AlphaZero tests must still pass.
  - Then N = 4, 30k games per member × 3 seeds, compared with the 30k-game AlphaZero self-play baseline.
- *Done when:* the decision to run or skip is recorded; if run, its results join Step 4.5's tables.

**Step 4.8 — Report section "Co-Evolution vs Self-Play"** — ✅ done
- **Why:** documents the result, including the case where self-play wins.
- **Objective & expectation:** a section in the same style as Phase 3, with H1–H7 explicitly confirmed or refuted.
- **Methodology:** assets from `scripts/make_report_assets.py`:
  - average and best member vs self-play by N (F1 and F2 side by side);
  - the held-out panel bars;
  - checkpoint curves;
  - diversity vs the self-play-seed baseline;
  - head-to-head heat-maps and cycle counts.
- *Done when:* the report compiles without undefined references and the new pages have been checked visually.

---

## Phase 5: Scale to 8x8 (~1.5 weeks)

**Design (decided 2026-10-07):**
- **Learner:** AlphaZero, the Phase 2 winner, with a residual network of **6 blocks × 64 channels** (6x6 Oroletto used 4 × 64).
- **Search:** **150 MCTS simulations per move** in self-play (6x6 used 64).
- **Budget:** **3 seeds × 100k self-play games**. Other settings follow 6x6, adapted only where board size requires it (e.g. the number of temperature plies).
- **External yardstick:** the open-source **Edax** engine, compiled from source, played at 2–3 strength levels, alongside re-tuned 8x8 heuristic and minimax bots.
- **Packaging:** after evaluation and level calibration, the model is packaged into the Othello-Player app (Step 5.8).
- **App levels (decided 2026-10-12):** only the **single strongest** 8x8 model is exported. The app's four levels use the same search budget and differ **only in move temperature** (no root noise). The values are calibrated in Step 5.7.
- **Run order (decided after measuring ~53 h per seed):** train **seed 0** first (~2.2 days). Then evaluate it (including Edax), calibrate the app's levels and package it into the app (Steps 5.4, 5.6–5.8). Seeds 1–2 train afterwards in the background (~4.4 days) to complete the 3-seed study (Steps 5.4–5.5, 5.9).

**Step 5.1 — Re-baseline on 8x8** — ✅ done (the best 8x8 weights equal the 6x6 ones; minimax-d4 beats the heuristic 99%; the heuristic beats random 93%)
Re-tune the heuristic weights and minimax depth for 8x8, then re-run the sanity gate from Step 0.11.

**Step 5.2 — Budget estimate** — ✅ done (measured ~470 s per 256-game iteration, ~53 h per seed including evaluation)
Use the throughput numbers to estimate compute before launching.

**Step 5.3 — Train the best configuration on 8x8** — seed 0 ✅ (100k games, 52.4 h; 96.6% vs minimax-d4); seeds 1–2 pending
Only the config should change (`board_size: 8`); there should be no code changes.

**Step 5.4 — Evaluate** — seed 0 ✅ (with 150 simulations: Edax level 1 98%, level 3 77%, level 5 42%, level 8 14%; minimax-d4 97.5%)
Against the 8x8 heuristic and minimax bots. Optionally, add an Edax subprocess bridge for ground-truth matches.

**Step 5.5 — Notes**
Record which 6x6 conclusions (Phases 2–4) held at 8x8 and which changed.

**Step 5.6 — Export the 8x8 model for the Othello-Player app** — ✅ seed 0 exported (ONNX matches PyTorch within 2.5e-6)
- **Why:** the app (separate repo `O-Rolleto-PlayOthello`, SPEC §4.2) is **8×8 only**. All four of its levels are played by this one model. There is **no distillation**.
- **Methodology:**
  - `scripts/export_onnx.py`: export the 8x8 network to ONNX (input `[batch, 3, 8, 8]`; outputs `policy_logits [batch, 65]` and `value [batch]`).
  - Check that ONNX and PyTorch outputs agree within 1e-5 on a few hundred positions.
  - Write a manifest with planes, c_puct, provenance (run id, commit) and the SHA-256 of the `.onnx` file.
  - Write golden fixtures for the app's parity tests: positions, network outputs, and greedy MCTS moves at fixed simulation counts.
- *Done when:* the `.onnx` file, the manifest and the fixtures are ready to copy into the app repo.

**Step 5.7 — Calibrate the app's four difficulty levels (8x8)** — ✅ verified 2026-10-13: Easy T=8, Medium T=2, Hard T=1, Master T=0 at 200 sims; adjacent levels 76%, 86%, 86% (all CIs above 70%)
- **Why:** the app plays every level with **one model, the strongest one**, weakened only by its **move temperature** (decided 2026-10-12). The temperatures must give clearly separated levels.
- **Methodology:**
  - **One search budget for all levels:** the largest simulation count whose worst-case move time stays within 2 s per move on this machine (single thread). Start at 200.
  - **Temperature T on visit counts:** the move is sampled with probability ∝ N^(1/T); T = 0 plays the most-visited move. No root noise.
  - Play a ladder of candidate temperatures T ∈ {0, 0.25, 0.5, 1, 2, 4} against each other (adjacent pairs, 400 games each), and each candidate against the 8x8 bots.
  - Pick four temperatures (Easy → Master, with Master T = 0) so that each level beats the one below at least 70% of the time.
- **Experiment details:** `configs/app_levels_8x8.yaml` (sims plus temperature placeholders), `scripts/eval_8x8.py levels`. Output: `results/eval_8x8/<run>/levels.json`. The chosen values go into the model manifest.
- *Done when:* the manifest records the shared `sims` and the four calibrated temperatures, with a calibration table.

**Step 5.8 — Package the model and link it into the Othello-Player app** — ✅ shipped seed 0 (app commit `b04b210`, levels calibrated in app commit `cad64e0`)
- **Why:** the app (`../Othello-Player`, GitHub `O-Rolleto-PlayOthello`) is 8×8 only and plays every level with **one model, the strongest**. Only that single `.onnx` is packaged. If a later seed (1–2) beats the packaged model head-to-head (95% CI excluding 50%), it replaces it.
- **Methodology:**
  - Copy the outputs of Steps 5.6–5.7 into the app repo, using the paths its SPEC expects:
    - `models/oroletto-8x8.onnx`;
    - `models/oroletto-8x8.json` (manifest: planes, c_puct, the shared `sims` and the calibrated per-level temperatures, provenance, SHA-256);
    - the parity fixtures under `tests/golden/`.
  - The files are copied, not symlinked, so the app builds on other machines.
  - Commit in the app repo with a short message. Don't push without asking.
  - Report anything the app's code still needs (ONNX inference, MCTS, move temperature from the manifest), without editing the app's code, which is developed separately.
- *Done when:* the files are committed in the app repo and the app's model-loading check (or a quick Python load of the `.onnx` file) passes.

**Step 5.9 — Report section "Scaling to 8×8"** — seed-0 version written (§8); add seeds 1–2 and verified levels later.
- **Why:** documents the 8x8 results, what changed from 6x6, and how the app's levels were calibrated.
- **Methodology:**
  - Assets generated from results files, as in every other section.
  - Prose following the experiment-report skill: motivation, methodology, results, then implications.
- *Done when:* the report compiles without undefined references and the pages have been checked visually.

---

## Phase 6: Exploiters & Blind-Spot Analysis — ❌ cancelled (2026-10-07)

Cancelled by the user before any work started. The design that was drafted (an exact endgame solver with a blunder audit of every agent; AlphaZero exploiters that model their victim, aimed at Oroletto, PPO and the best Phase 3 DQN; a transfer matrix; exploitability of league and co-evolved models) is kept in git history (commit `1a50d6b`) in case it is revisited.

---

## Phase 7: Variable Board-Size Generalization (~2.5 weeks) — ⏸ DEFERRED (2026-10-15, user decision; not run)

> Note for when it is picked up: the current networks have size-specific heads (flatten layers), so Step 7.1 needs the size-agnostic network from 7.2 trained on one size to give a real zero-shot result. Open decisions: sizes, seeds, baseline run, app shipping.

**Step 7.1 — Zero-shot baseline**
Evaluate the 8x8 network on 6x6 and 10x10 and document the failure. If the network can't even run on those sizes because of a flatten layer, that counts as part of the result.

**Step 7.2 — Size-agnostic network**
A fully convolutional policy head (1x1 conv → one logit per square, plus a pass logit from pooled features) and global average pooling before the value head.
*Done when:* the same weights run a forward pass on sizes 6, 8 and 10.

**Step 7.3 — Multi-size self-play**
Sample the board size per game, and group batches by size (or pad and mask).

**Step 7.4 — Mixed training run**
Train on 6, 8 and 10 together.

**Step 7.5 — Curriculum run**
6 → add 8 → add 10, using the same total compute as Step 7.4.

**Step 7.6 — Train × test matrix**
Win rate of each model (8x8-only, mixed, curriculum) on each test size, against that size's heuristic bot. **This is the headline result.**

---

## Final: Writeup & Add-ons (~1 week)

**Step F.0 — Combined Elo rating of every agent** *(the only place Elo is used)*
- **Why:** a single strength scale for the whole project, placing every agent from every phase side by side.
- **Methodology:**
  - Fit Bradley–Terry Elo (`othello/eval/elo.py`) on 6x6, with random = 0.
  - Players: the rule-based bots, the Phase 1 agents, the Phase 2 agents, the Phase 3 league DQNs, and the Phase 4 population models.
  - Data: existing results files plus any extra games needed to connect the comparison graph (e.g. a minimax-d6 anchor above Oroletto).
  - Check: every predicted score must fall within the observed score's confidence interval. Agents with large residuals, such as opponent-exploiting MC control, are flagged.
- *Done when:* the Elo table and chart, with the consistency check, are in the final report.

**Step F.1 —** Pick 1–2 add-ons: symmetry-augmentation ablation (cheap, reuses `symmetry.py`), a compute budget table, or a value-net linear probe.
**Step F.2 —** Regenerate every plot and table from the logged CSVs with scripts, so no numbers are entered by hand.
**Step F.3 —** Write the final report.

---

## Rules applied at every step
- `pytest tests/` passes before any training run, and the env tests cover sizes 6, 8 and 10.
- One evaluation protocol everywhere: both colors, randomized openings, fixed eval seeds, confidence intervals.
- Every run is reproducible from its config and seed.
