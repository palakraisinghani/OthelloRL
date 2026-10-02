# Othello RL Project: Full Pipeline (Starting on 6x6)

## Phase 0 — Environment & Infrastructure

**Goal:** One reusable codebase for every later phase.

1. Build a size-parameterized Othello environment (`board_size` as a constructor argument, not hardcoded) — even though you're starting at 6x6, write it so 8x8/10x10 need zero code changes.
2. Implement: legal move generation, disc-flipping logic, forced-pass handling, terminal/win detection.
3. Build two fixed reference opponents:
   - **Random-move bot**
   - **Heuristic bot** (weighted sum of disc count + mobility + corner occupancy — tune weights briefly so it's a genuinely non-trivial opponent)
4. Build an **alpha-beta minimax bot** at a fixed shallow depth (e.g., 4–6 ply) — your non-RL ceiling reference.
5. Build eval harness: play N games between any two agents, report win/draw/loss rate, log game replays.
6. Sanity check: heuristic bot should beat random bot consistently; minimax should beat heuristic bot consistently. Confirms your environment logic is correct before any RL touches it.

**Deliverable:** working 6x6 env + 3 baseline bots + eval harness.

---

## Phase 1 — Traditional Methods (and why they fail)

**Goal:** Empirically demonstrate why naive RL doesn't work here, on 6x6, against your fixed heuristic bot.

Train and evaluate each of these, same opponent, same number of training games, same eval protocol:

1. **Tabular Q-learning**
2. **SARSA**
3. **First-visit Monte Carlo control**
4. **Linear function approximation Q-learning** (features: disc count, mobility, corner occupancy, edge stability)
5. **Vanilla DQN** (raw board input, no self-play sophistication, small MLP or CNN)
6. **Alpha-beta minimax** (already built in Phase 0 — included here as the non-learning reference point)

**Key experiment:** win-rate vs. heuristic bot, plotted against training episodes, all 5 learning methods on one chart.

**Key ablation:** for method 4, run once with disc-count-heavy reward/features and once without, to explicitly show the "maximizing disc count is a trap" phenomenon.

**Why each fails (for the writeup):**
- Tabular Q-learning / SARSA / MC control: state space too large to enumerate even on 6x6 — states are rarely revisited, so the table never fills in (curse of dimensionality).
- MC control specifically: high-variance updates since returns only arrive at game end.
- Linear function approximation: "half-works" — beats random play but plateaus hard; disc-count-heavy variants can actively hurt performance since maximizing disc count mid-game is often a losing strategy in Othello.
- Vanilla DQN: trains but plateaus below a good heuristic bot due to lack of self-play sophistication and search-based correction.

**Deliverable:** failure-analysis plot + writeup of *why* each method plateaus/fails.

---

## Phase 2 — Three Modern Methods

**Goal:** Show self-play + better algorithms clear the Phase 1 ceiling.

Still on 6x6, evaluated against the same fixed heuristic/minimax bots (training itself is now self-play, not against the fixed bot):

1. **Self-play PPO** (actor-critic, no search)
2. **Self-play DQN+** (Double DQN, replay buffer, opponent pool of past checkpoints rather than pure latest-self-play)
3. **AlphaZero-style** (small ResNet, dual policy/value head, MCTS + self-play, legal-move masking, dihedral symmetry augmentation)

**Comparison table:** final win-rate vs. heuristic/minimax bot, sample efficiency (games to reach a threshold win-rate), wall-clock time, training stability (variance across seeds).

**Deliverable:** trained checkpoints for all 3 methods + comparison table/plot. Pick the strongest (likely AlphaZero-style) to carry forward.

---

## Phase 3 — Architecture Sweep

**Goal:** Rigor around your best method from Phase 2, still on 6x6.

Fix algorithm = winner of Phase 2. Vary one axis at a time:

1. Depth (residual blocks: 2 / 4 / 6 / 10)
2. Width (channels: 32 / 64 / 128)
3. Input planes (mine/theirs only → + empty-square plane → + turn indicator → + legal-move mask)
4. Head design (shared trunk vs. separate trunks for policy/value)
5. MCTS simulation budget (25 / 100 / 400 sims per move, network fixed)

**Deliverable:** grid/table of strength (Elo or win-rate vs. minimax) vs. compute cost (params / FLOPs / time) — a compute/strength Pareto frontier plot.

---

## Phase 4 — Scale to 8x8

**Goal:** Confirm your best architecture (from Phase 3) and training recipe transfer to standard tournament-size Othello, using the same size-agnostic environment from Phase 0.

1. Re-run your winning Phase 2/3 configuration on 8x8 (expect longer training — budget accordingly).
2. Re-validate against heuristic/minimax bots built for 8x8.
3. Optional: compare against a published/known solver (e.g., Edax) for a ground-truth sanity check, since 8x8 has this resource available (6x6 mostly doesn't).

**Deliverable:** trained 8x8 agent + confirmation that Phase 3's architecture choices hold at the larger size (or note what changed).

---

## Phase 5 — Variable Board Size Generalization

**Goal:** One agent, multiple board sizes — your capstone result.

1. **Naive baseline check:** take the 8x8-trained network, zero-shot eval on 6x6/10x10 — expect failure, document it.
2. **Make the architecture size-agnostic:** fully convolutional policy head (spatial softmax over legal moves) + global average pooling before the value head, so no layer hardcodes board dimensions.
3. **Multi-task training:** train one network on a mixture of board sizes (6x6, 8x8, 10x10) sampled into the same training loop.
4. **Curriculum variant:** train small→large progressively (6x6 first, then introduce 8x8, then 10x10), compare against the mixed-training approach from step 3.
5. **Evaluation:** train-size × test-size matrix of win-rate vs. each size's heuristic bot — this table is your headline result.

**Deliverable:** size-agnostic network + train×test win-rate matrix + writeup comparing naive/multi-task/curriculum approaches.

---

## Optional Add-Ons (if time allows, pick 1–2)

- **Interpretability probe** on the value network (Othello-GPT style) — is board state linearly decodable from activations?
- **Symmetry augmentation ablation** — sample efficiency with vs. without 8-fold dihedral augmentation.
- **Failure gallery** — annotated losing games from your weakest method(s), for the writeup.
- **Compute budget table** — GPU-hours per phase, to justify board-size choices and frame the Pareto tradeoffs.

---

## Timeline Sketch

| Phase | Content | Est. Time |
|---|---|---|
| 0 | Env + baselines | ~1 week |
| 1 | Traditional methods (6x6) | ~1.5 weeks |
| 2 | PPO / DQN / AlphaZero (6x6) | ~2.5 weeks |
| 3 | Architecture sweep (6x6) | ~1 week |
| 4 | Scale to 8x8 | ~1.5 weeks |
| 5 | Variable-size generalization | ~2.5 weeks |
| — | Writeup, ablations, polish | ~1 week |
