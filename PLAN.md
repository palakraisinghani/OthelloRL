# Othello RL: Step-wise Plan

The steps implement `othello_rl_project_roadmap.md`. Do them in order. Each step has a **Done when** check, and you only move on once it passes.

**Default stack:** Python 3.11, NumPy for the env, PyTorch for the networks, pytest, matplotlib. Each run uses one YAML config, writes its logs to `results/` as CSV/JSONL, and has its seeds fixed.

---

## Phase 0: Environment & Baselines (~1 week)

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
`encode(state, planes=[...])` with configurable planes: mine, theirs, empty, turn, legal mask. Phase 3 sweeps over these.
*Done when:* the output shape is `(C, n, n)` for each plane config and each size.

**Step 0.6 — Agent interface + Random bot**
`Agent.act(state) -> action` in `agents/base.py`; `agents/random.py`.

**Step 0.7 — Heuristic bot**
Score = disc count + mobility + corner occupancy, using position weights generated from the board geometry (corners, X/C-squares, edges). Use random tie-breaking or a small ε so the bot isn't fully deterministic, which stops tabular methods from memorizing a single line against it.

**Step 0.8 — Alpha-beta minimax bot**
Depth 4–6, uses the heuristic as its evaluation, with move ordering.

**Step 0.9 — Eval harness**
`eval/arena.py`: plays N games between any two agents with colors swapped, starting from K randomized opening positions, using multiprocessing. Reports W/D/L with confidence intervals. `eval/replay.py` saves and prints games. `eval/elo.py` computes round-robin Elo.

**Step 0.10 — Tune heuristic weights**
Run a short grid or random search over the weights, playing against random and against earlier weight settings.

**Step 0.11 — Sanity gate**
*Done when:* on 6x6 over 1,000 games, heuristic beats random ≥95% and minimax beats heuristic ≥80%. Record env throughput (games/sec); it is needed for compute budgeting later.

---

## Phase 1: Traditional Methods on 6x6 (~1.5 weeks)

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

## Phase 2: Modern Self-Play Methods on 6x6 (~2.5 weeks)

**Step 2.1 — Opponent pool**
A shared module that saves checkpoint snapshots and samples opponents from them. It is used by both DQN+ and PPO.

**Step 2.2 — Self-play DQN+**
Extend `dqn.py` with Double DQN and opponent-pool self-play.

**Step 2.3 — Self-play PPO**
`rl/ppo.py`: shared actor-critic, masked softmax, GAE, opponent pool.

**Step 2.4 — AlphaZero: network**
`rl/alphazero/net.py`: small ResNet with policy and value heads.

**Step 2.5 — AlphaZero: MCTS**
`mcts.py`: PUCT, Dirichlet noise at the root, temperature schedule, legal-move masking. Batch the network calls from the start, because Phases 3–5 depend on MCTS being fast.
*Done when:* MCTS with a dummy uniform network beats random.

**Step 2.6 — AlphaZero: self-play + training loop**
`selfplay.py` (parallel workers) and `train.py` (replay window, periodic eval/gating), plus `symmetry.py` for 8-fold dihedral augmentation.

**Step 2.7 — Run all 3 methods**
3 seeds each, evaluated against heuristic and minimax.

**Step 2.8 — Comparison table**
Columns: final win rate, games to reach the threshold (e.g. 80% vs heuristic), wall-clock time, and variance across seeds. Also run round-robin Elo across every agent built so far. As an extra check, 6x6 is weakly solved (perfect play ends 16–20, a second-player win), so compare the best agent's self-play against that result.
*Done when:* the table and checkpoints are saved and a winner is chosen.

---

## Phase 3: Architecture Sweep on 6x6 (~1 week)

**Step 3.1 — Sweep infrastructure**
`scripts/sweep.py` takes a baseline config plus a single axis override. Log params, FLOPs and training time.

**Step 3.2 — Depth sweep:** 2 / 4 / 6 / 10 residual blocks.
**Step 3.3 — Width sweep:** 32 / 64 / 128 channels.
**Step 3.4 — Input-plane sweep:** mine/theirs → +empty → +turn → +legal mask.
**Step 3.5 — Head design:** shared trunk vs. separate trunks.
**Step 3.6 — MCTS sims:** 25 / 100 / 400 at eval time, with the network fixed.

**Step 3.7 — Pareto plot**
Plot strength (Elo / win rate vs minimax) against compute.
*Done when:* the plot is made and the best configuration is chosen.

---

## Phase 4: Scale to 8x8 (~1.5 weeks)

**Step 4.1 — Re-baseline on 8x8**
Re-tune the heuristic weights and minimax depth for 8x8, then re-run the sanity gate from Step 0.11.

**Step 4.2 — Budget estimate**
Use the throughput numbers to estimate GPU-hours before launching.

**Step 4.3 — Train the best configuration on 8x8**
Only the config should change (`board_size: 8`); there should be no code changes.

**Step 4.4 — Evaluate**
Against the 8x8 heuristic and minimax bots. Optionally, add an Edax subprocess bridge for ground-truth matches.

**Step 4.5 — Notes**
Record which Phase 3 conclusions held at 8x8 and which changed.

---

## Phase 5: Variable Board-Size Generalization (~2.5 weeks)

**Step 5.1 — Zero-shot baseline**
Evaluate the 8x8 network on 6x6 and 10x10 and document the failure. If the network can't even run on those sizes because of a flatten layer, that counts as part of the result.

**Step 5.2 — Size-agnostic network**
A fully convolutional policy head (1x1 conv → one logit per square, plus a pass logit from pooled features) and global average pooling before the value head.
*Done when:* the same weights run a forward pass on sizes 6, 8 and 10.

**Step 5.3 — Multi-size self-play**
Sample the board size per game, and group batches by size (or pad and mask).

**Step 5.4 — Mixed training run**
Train on 6, 8 and 10 together.

**Step 5.5 — Curriculum run**
6 → add 8 → add 10, using the same total compute as Step 5.4.

**Step 5.6 — Train × test matrix**
Win rate of each model (8x8-only, mixed, curriculum) on each test size, against that size's heuristic bot. **This is the headline result.**

---

## Final: Writeup & Add-ons (~1 week)

**Step 6.1 —** Pick 1–2 add-ons: symmetry-augmentation ablation (cheap, reuses `symmetry.py`), a failure gallery (from `replay.py`), a compute budget table, or a value-net linear probe.
**Step 6.2 —** Regenerate every plot and table from the logged CSVs with scripts, so no numbers are entered by hand.
**Step 6.3 —** Write the final report.

---

## Rules applied at every step
- `pytest tests/` passes before any training run, and the env tests cover sizes 6, 8 and 10.
- One evaluation protocol everywhere: both colors, randomized openings, fixed eval seeds, confidence intervals.
- Every run is reproducible from its config and seed.
