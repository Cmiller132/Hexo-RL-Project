# Tuning Orchestrator Guide

Date: 2026-04-28

This is a grounding memo for an autonomous GPT-5.5 medium orchestrator that
wakes every 30 minutes to watch over ASHA/BOHB/PB2 tuning.

The orchestrator is expected to improvise. This document exists to preserve the
goal, context, and important failure modes after many context compactions.

## Goal

Produce a few well-tuned, bug-clean models from different architecture families
that show competent Hexo strategy and can threaten or beat the classical bot.

The run is successful only if the models are strong for real reasons:

- legal games;
- correct replay reconstruction;
- clean D6 symmetry behavior;
- correct target generation;
- no target leakage;
- no missing legal actions;
- no hidden fallback-prior dependence;
- tactical competence, not just long passive losses.

Do not keep a broken run alive just because a metric looks good.

## Required Context

Use these docs as the source of truth:

```text
Docs/game.md
Docs/SPEC_FIX_MATCH_PHASE1_PHASE3_COMPLETION_20260428.md
Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md
Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md
Docs/RGSC_IMPLEMENTATION.md
Docs/MODEL_HEAD_TARGET_AND_D6_FIXES_20260428.md
```

Important game facts:

- Player 0 opens with one stone at origin.
- After that, turns are two placements by the same player.
- 4-windows and 5-windows are both immediate win-in-one-turn threats.
- Strong play is mostly about building multiple threats across axes so two
  defensive placements cannot cover them.
- D6 means all 12 hex symmetries.
- Humans and game players can place anywhere legally; threat restriction is a
  training/search device, not the game rule.

## Wake Loop

Every 30 minutes:

1. Check whether the supervisor and workers are alive.
2. Read newest events, trial scorecards, warnings, and dashboard/run DB state.
3. Check hard bug sentinels before judging strength.
4. Check resource health: GPU, CPU, RAM, swap, disk, shared memory, timeouts.
5. Check training health: losses finite, targets present, enabled heads trained.
6. Check self-play health: positions/min, game length, terminal reasons,
   fallback prior use, legal counts, MCTS/inference latency.
7. Check evaluation health: tactical suites, classical survival, checkpoint
   league, outside-window tests.
8. Decide whether to observe, retune, quarantine, patch, restart, or stop.
9. Write a short note: status, best trials, problems, actions, next watch item.

## Hard Sentinels

These override Elo, survival, losses, and tournament results:

```text
illegal_move_rate
post_terminal_move_attempts
replay_mismatch_rate
d6_mismatch_rate
legal_mask_mismatch_rate
oracle_threat_mismatch_rate
missing_legal_action_rows
pair_mask_violation_rate
target_leakage_check_status != pass
non_finite_loss
non_finite_model_output
checkpoint load failure for a promoted trial
```

If any hard sentinel fires, quarantine the trial/family, preserve evidence, make
a small repro, fix the root cause, and only resume after tests pass.

## Scoring Intuition

Do not overread early models.

```text
epoch < 8:
  mostly random
  score health, target learning, finite losses, throughput, and bug cleanliness
  ignore classical survival as a strength signal

8 <= epoch < 12:
  maybe strategy shape is forming
  score tactical fixtures, outside-window diagnostics, value/policy quality,
  and bug cleanliness

epoch >= 12:
  classical survival starts to mean something
  long losses can indicate structure, but they are not wins

epoch >= 20:
  checkpoint league and tactical holdouts should dominate champion decisions
```

Classical survival is useful, especially because weak models may rarely win.
Surviving to 80-100 moves against classical is meaningfully better than losing
before 20-30 moves, but it can be gamed by passive delay. Trust it only
alongside tactical progress and clean games.

Final strength should come from checkpoint league lower-confidence bound,
tactical/outside-window suites, and real games viewed in the dashboard.

## Tuning Methods

ASHA narrows static choices:

```text
architecture
model size
sim count
head bundle
candidate/context-token settings
pair-policy mode
runtime recipe
```

Default ASHA should reach meaningful rungs before pruning:

```text
8,12,14
```

BOHB should be ASHA/Hyperband plus model-based sampling. If it is just random
search plus rungs, call it that.

PB2 should tune dynamic schedules with a response model and uncertainty-aware
proposals:

```text
LR, weight decay, c_puct, Dirichlet noise, PCR probability, replay freshness,
loss weights, sparse/pair/regret weights
```

If it is just clone-and-random-mutate, call it PBT fallback.

## Model Families To Keep Alive

Try to preserve architecture diversity long enough to get real signal:

```text
best_current_33
restnet_crop_scout
candidate_policy_33
graph_hybrid_0
global_xattn_0
global_line_window_0
global_pair_twostage_0
global_graph_full_0
global_hybrid_action_0, if cheap enough
global_graph768_champion, only after smaller graph designs pass gates
```

Do not let one fast family erase all others before the bug gates and meaningful
strategy rungs have run. Do quarantine a family that is actually broken.

## Likely Bugs

Watch especially for:

- turn-boundary mistakes from forgetting the one-stone opening and two-stone
  turns;
- value/lookahead targets mixing player perspectives;
- opponent policy accidentally targeting the same player's second placement;
- low-PCR policy targets training the policy head;
- truncated games training terminal value or regret;
- sparse/graph/pair D6 silently disabled or wrong;
- candidate or graph features leaking target probability/future labels;
- missing legal action rows in global graph;
- pair policy active on opening or using the wrong second-placement legal set;
- fallback priors dominating MCTS top-k;
- enabled heads with no target or no loss weight;
- subtree reuse silently disabled or breaking re-root;
- late-game high-legal-count positions causing inference timeouts.

## When To Intervene

Use the smallest useful intervention.

Observe when metrics are noisy but healthy.

Retune when the system is healthy but inefficient:

- workers;
- batch sizes;
- inference wait;
- sim counts;
- LR/noise/search ranges;
- eval games.

Quarantine when one trial/family repeatedly violates hard gates or timeouts.

Patch when there is a reproducible bug. Add or update the test first when
practical, then fix the root cause.

Restart only after preserving logs/checkpoints/replay examples, or when the
process is unrecoverable.

Stop the campaign if the data may be corrupted or hard sentinels are unreliable.

## What To Tune First

If bug-clean but weak:

1. Search: sims, `c_puct`, Dirichlet noise, PCR schedule, subtree reuse.
2. Training: LR, weight decay, batch size, replay freshness, value/policy
   balance.
3. Aux heads: pair, opponent policy, lookahead, regret, axis weights.
4. Architecture: graph family, token/context design, pair-policy mode,
   relation depth.
5. Evaluation: more league games, stronger tactical suites, fixed openings.

Do not tune architecture around a broken data/search contract.

## Signs Of Real Progress

Real improvement should show up in several places:

- losses improve without heads silently dropping out;
- top policy mass becomes less random in high-legal-count states;
- tactical fixtures improve;
- classical losses get longer after epoch 12;
- checkpoint league LCB improves;
- dashboard replays show multi-axis structure and better blocking;
- D6/replay/oracle sentinels remain clean.

Suspicious improvement:

- better survival but no tactical improvement;
- wins only with fallback priors;
- strength only from one evaluator;
- large gains right after a data-contract change;
- graph model wins while missing legal rows or pair masks are nonzero.

## Visual Match Snapshots

Use PNG snapshots to sanity-check whether strategy is actually developing.
The dashboard exposes:

```text
GET /api/games/{game_id}/snapshot.png?run_id={trial_id}
```

The CLI equivalent is:

```text
python scripts/render_match_png.py --suite-root runs/<suite> --run-id <trial_id> --latest 4
```

Default snapshots fit around played stones, not the full legal cloud. This makes
classical-vs-NN structure easier to compare: look for coherent axis building,
fork attempts, forced blocks, repeated failure to block, passively scattered
stones, and late-game policy collapse into random-looking placements.

## Success State

End with:

- at least one bug-clean crop/CNN-family reference;
- at least one bug-clean `graph_hybrid_0` or candidate-policy reference;
- at least one bug-clean global graph alternative;
- a champion candidate with competent strategy against classical;
- checkpoint league evidence with confidence;
- tactical and outside-window suite evidence;
- replayable dashboard games;
- persisted configs, checkpoints, scorecards, and monitor notes explaining why
  the promoted models were selected.

Competent strategy means visible structure: multi-axis development, useful
blocking, longer classical survival after epoch 12, tactical-suite competence,
and no reliance on illegal moves, replay bugs, D6 bugs, target leakage, or
fallback-prior artifacts.
