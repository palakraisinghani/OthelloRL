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
*Done when:* the table and checkpoints are saved and a winner is chosen. The winner is released as **Oroletto** (may be replaced by a stronger Phase 5 model): point `models/oroletto.json` at its run directory so `othello --play oroletto` loads it.

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

## Phase 4: Exploiters & Blind-Spot Analysis (6x6, ~1.5 weeks)

**Question:** where are our agents' blind spots, and how exploitable is each one? Targets are all 11 agents: 3 rule-based, 5 traditional, and the Phase 2 finals (DQN+, PPO, Oroletto), plus the Phase 3 league-trained DQN models.
**⚠ Design to be discussed before starting.**
- **Exploiter type:** proposed default is an AlphaZero exploiter whose search models the victim with the victim's own network (A-MCTS, Wang et al. 2023). Alternatives are a PPO exploiter, or both.
- **Analysis scope:** proposed default is an exact-solver audit of all agents, plus exploiters trained only against the strong agents.

**Step 4.0 — Generic batched game runner** *(moved here from Phase 3)*
- **Why:** AlphaZero exploiters (4.4) and AZ-vs-AZ or AZ-vs-agent transfer matches (4.5) need a game loop where either side can be an MCTS player or a plain agent. Phase 5's populations need it too.
- **Methodology:** `run_games(games)` in `othello/rl/alphazero/selfplay.py`. Each step groups games by the player to move, plain agents move directly, and each MCTS player runs one batched search. `az_evaluate` and self-play become thin wrappers.
- *Done when:* the existing AlphaZero tests pass, and new tests cover MCTS-vs-Agent and MCTS-vs-MCTS.

**Step 4.1 — Exact endgame solver**
Alpha-beta search on the exact final disc margin, with a transposition table and move ordering. It solves any 6x6 position with ≤ E empty squares; E (~12–14) is set by timing.
*Done when:* it matches brute force on 4x4 games and on random 6x6 positions with few empties, and the time per solve is recorded.

**Step 4.2 — Blunder audit**
Each agent plays the same set of openings against a shared panel of opponents. At every position with ≤ E empties, the solver labels the agent's move as:
- **optimal**;
- **inaccurate:** it loses discs but not the result;
- **a blunder:** it changes the game-theoretic result, from win to draw or loss, or from draw to loss.

*Done when:* `results/blindspots/audit_<agent>.jsonl` exists for all agents, with per-move labels and position features.

**Step 4.3 — Blind-spot taxonomy**
Describe each audited position by:
- game phase (number of empties);
- the square class of the move and of the best move (corner, X, C, edge, interior);
- the number of winning moves available (how hard the position is);
- pass situations;
- parity of empty regions;
- the mobility difference.

Compare blunder rates per category and per agent, normalised per move.
*Done when:* there is a table and figure of blunder rate by category × agent.

**Step 4.4 — Exploiter training**
Train an exploiter against a frozen Oroletto, which plays with its evaluation settings, and against Phase 2 PPO and the best Phase 3 DQN model. Save checkpoints.
*Done when:* each exploiter's win rate against its victim is logged over training.

**Step 4.5 — Exploitability and transfer**
Each exploiter plays all 11 agents (and the Phase 3 models): a transfer matrix of exploiter × target. Do one model's weaknesses carry over to the others?
*Done when:* the matrix is produced, with confidence intervals.

**Step 4.6 — What do exploiters exploit?**
Run the 4.2 audit on the victim's moves in exploiter games and compare them with its moves in ordinary games:
- which categories become over-represented;
- which openings and positions the exploiter steers into.

Build an annotated gallery of the most instructive losses (from `replay.py`).
*Done when:* the main blind spots are named and illustrated with board diagrams.

**Step 4.7 — Link back to Phase 3**
Are league-trained DQN models less exploitable than the self-play DQN (p = 0)?
*Done when:* an exploitability comparison table exists.

**Step 4.8 — Report section "Blind Spots and Exploitability"**
*Done when:* it compiles and the pages have been checked.

---

## Phase 5: Multi-Agent Co-Evolution vs Self-Play (6x6, ~1.5 weeks)

**Question:** do several *independent* agents (separate networks, separate learners), learning by playing each other, end up better or more robust than agents trained by self-play?
**⚠ Design to be discussed before starting.** Decisions:
- **population size N** (proposed: 4);
- **homogeneous vs heterogeneous algorithms**, e.g. all-AlphaZero vs a PPO + DQN+ + AlphaZero mix (proposed: try PPO populations first because they are cheap, then AlphaZero);
- **matchmaking** (proposed: uniform among the other members; strength-matched by recent win rate as a variant);
- **budget fairness** (proposed: report both equal games per agent and equal total compute).

**Step 5.1 — Population trainer**
N agents, each with its own network, optimiser and replay buffer. Every training game pairs two *different* members, so there are no games against itself. Each member learns only from its own moves. It reuses the Phase 4 `run_games` runner and the Phase 3 League sampler (as a matchmaker), and saves checkpoints for every member.
*Done when:* a smoke run with N = 2 trains both members and logs the pairwise results.

**Step 5.2 — PPO populations**
N ∈ {2, 4} (8 if cheap enough) × 3 seeds, with the same per-agent game budget as Phase 2 PPO.
*Done when:* all runs finish with checkpoints.

**Step 5.3 — AlphaZero population**
N = 4, 30k games per agent × 3 seeds, then confirm at full budget if it is promising.
*Done when:* all runs finish.

**Step 5.4 — Comparison with self-play**
For each member and for the population's best:
- head-to-head against the self-play model at the same budget;
- matches against held-out anchors (minimax-d6, Oroletto);
- exploitability, using the Phase 4 exploiters (transfer).

*Done when:* there is a comparison table at equal per-agent games and at equal total compute.

**Step 5.5 — Population dynamics**
- Track each member's strength over time: its checkpoints' scores against fixed anchors (minimax-d2/d3/d4, Oroletto) and a win-rate matrix between checkpoints.
- Check for **non-transitivity**: count rock-paper-scissors cycles among checkpoints (A > B > C > A).
- Measure **diversity** between members: how often their policies disagree on a fixed set of positions.

Does co-evolution produce cycling or real progress?
*Done when:* there are figures for strength over time, the cycle count and diversity.

**Step 5.6 — Report section "Co-Evolution vs Self-Play"**
*Done when:* it compiles and the pages have been checked.

---

## Phase 6: Scale to 8x8 (~1.5 weeks)

**Step 6.1 — Re-baseline on 8x8**
Re-tune the heuristic weights and minimax depth for 8x8, then re-run the sanity gate from Step 0.11.

**Step 6.2 — Budget estimate**
Use the throughput numbers to estimate compute before launching.

**Step 6.3 — Train the best configuration on 8x8**
Only the config should change (`board_size: 8`); there should be no code changes.

**Step 6.4 — Evaluate**
Against the 8x8 heuristic and minimax bots. Optionally, add an Edax subprocess bridge for ground-truth matches.

**Step 6.5 — Notes**
Record which 6x6 conclusions (Phases 2–5) held at 8x8 and which changed.

---

## Phase 7: Variable Board-Size Generalization (~2.5 weeks)

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
  - Players: the rule-based bots, the Phase 1 agents, the Phase 2 agents, the Phase 3 league DQNs, and the Phase 5 population models.
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
