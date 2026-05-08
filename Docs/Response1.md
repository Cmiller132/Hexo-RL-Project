## Verdict

This is a **serious, coherent implementation design**, and it fixes many of the failure modes I was worried about in the earlier architecture: Rust owns legality and tree search, Python only supplies bounded candidate reservoirs and neural scores, replay explicitly avoids treating unsampled legal pairs as negatives, and interior nodes are now genuinely recursive pair-MCTS nodes rather than root-only pair scoring. Those are major improvements. 

But I see several issues that could make the implementation **look like V1 pair MCTS while not actually testing the intended hypothesis**. The biggest ones are:

1. The described root search may not actually implement **Gumbel sequential halving**; it appears closer to ordinary PUCT with a tiny Gumbel tie-break.
2. The two-pass neural evaluation per expanded node may violate the intended performance model unless the trunk is cached or batched aggressively.
3. Terminal one-stone wins inside unordered pairs create an ordering/semantics ambiguity.
4. The policy target is still underspecified relative to the biases introduced by candidate admission, Gumbel admission, progressive widening, and visit-count selection.
5. The implementation validates legal-row identity, but it may also need a stronger **full state identity hash** to prevent Python/Rust graph drift.
6. Progressive widening with priors normalized over unrevealed candidates can distort PUCT unless priors are renormalized over the currently revealed set.

My opinionated take: **this is good enough to prototype, but not good enough to accept as “the V1 pair-search implementation” until the root Gumbel behavior, target construction, terminal-pair semantics, and batching/performance contracts are tightened.**

---

## What is strong

The split of authority is right. Rust owns legal-row identity, pair application, traversal, PUCT, progressive widening, backup, and final move application; Python owns model inference and candidate construction. That is the correct boundary for a system where silent legality drift would poison search and replay. 

The implementation also keeps the important V1 distinction between **candidate generation** and **final decision-making**. Python builds a bounded candidate reservoir from learned and tactical sources, then Rust searches over admitted pair rows; the selector is not allowed to directly choose the final move. That matches the architecture’s non-negotiable contract that final root choice must be a canonical unordered pair action, not a pair-to-single projection or threat-filtered action.  

The recursive expansion lifecycle is also a real step forward. Earlier designs can accidentally become “root pair scorer plus shallow rollout.” This description has Rust requesting interior expansions, Python scoring interior candidate reservoirs with the same pipeline as the root, and Rust caching reservoirs for later PUCT selection and widening. That is actual recursive pair-action MCTS. 

The replay/training language is mostly correct: admitted pairs can receive policy/ranking/Q targets, unsampled legal pairs are not implicit negatives, tactical labels come from Rust tactical payloads rather than candidate masks, and runtime-only legal projection tensors are ignored by the loss planner. That is exactly the right statistical stance for sampled pair search. 

The strongest implementation choice is probably the hard failure behavior. Missing value outputs, threat-filtered legal rows, stale generation tokens, duplicate/illegal pairs, non-finite values/logits, stale hashes, and pending-expansion selection all fail loudly. That is the right default for a new cross-language search/runtime schema. 

---

## Major issue 1: the described “Gumbel” root behavior may not be Gumbel policy improvement

This is the most important concern.

The implementation says Rust stores deterministic Gumbel values, and then root PUCT uses:

```text
score = Q(root_edge)
      + c_puct * prior(root_edge) * sqrt(parent_visits)
        / (1 + edge_visits)
      + tiny_gumbel_tiebreak
```

That is not the same thing as Gumbel AlphaZero / Gumbel MuZero-style policy improvement. It is basically PUCT with a tiny Gumbel tie-break. The architecture explicitly called for `gumbel_sequential_halving_v1` root admission with admitted set, Gumbel values/admission order, simulation allocation, completed-Q values, and an improved policy target over admitted candidates. 

The distinction matters. Gumbel AlphaZero was proposed specifically because AlphaZero-style training can fail to improve the policy network when not all root actions are visited; it samples actions without replacement and uses a different policy-improvement procedure, especially useful with few simulations. ([OpenReview][1])

So I would not let the implementation call this “Gumbel sequential halving” unless it explicitly does something like:

```text
1. sample/admit a root considered set using logits + Gumbel noise;
2. allocate simulations by sequential halving or documented Gumbel root schedule;
3. compute completed-Q values for considered root actions;
4. build the replay target from the Gumbel improved policy, not raw visits alone;
5. expose admission order, considered set, completed-Q, and allocation in replay.
```

Right now, the description sounds like:

```text
candidate reservoir -> softmax prior -> PUCT -> tiny Gumbel tie-break
```

That is a valid search variant, but it is not the Gumbel policy-improvement variant the architecture seemed to want.

---

## Major issue 2: PUCT has a zero-visit first-selection problem

The PUCT formula uses:

```text
sqrt(parent_visits)
```

At a newly expanded root or interior node, `parent_visits` / `node_visits` can be zero. That makes the exploration term zero for every action:

```text
U = c_puct * prior * sqrt(0) / (1 + edge_visits) = 0
```

Then the first selection is governed by Q initialization and tie-breaks, not by the prior.

This is not necessarily fatal; many MCTS variants tolerate a special first visit. But here it matters because the action set is a sampled candidate set and because the system relies heavily on learned priors. I would change the formula to use:

```text
sqrt(max(1, parent_visits))
```

or define a specific first-selection rule:

```text
first revealed edge selection = argmax prior_logit + gumbel
```

The implementation already uses `max(1, node_visits)` for progressive widening, so using the same convention in PUCT would be consistent. 

---

## Major issue 3: terminal pair ordering is ambiguous

The implementation says selected pairs are applied atomically by applying the first placement, checking terminal, then applying the second placement and checking terminal. But the V1 action is an **unordered canonical pair**. 

That creates a subtle but important issue for 5-window wins. In Hexo, a 5-window has one empty cell; filling it wins, and the other placement is strategically irrelevant filler. The game guide explicitly says 5-windows and 4-windows are both immediate win-in-one-turn threats because each turn gives two placements. 

Suppose the unordered pair is:

```text
{winning_cell, filler_cell}
```

If the canonical order applies `winning_cell` first, the game ends before `filler_cell` is placed. If the canonical order applies `filler_cell` first, then `winning_cell`, the terminal result is the same, but the terminal move history/board trace contains a strategically arbitrary filler stone.

For search value, both are equivalent. For replay and policy targets, this can become toxic: the system may learn arbitrary filler preferences. The architecture already warns that terminal-equivalent targets must not train arbitrary filler-stone preferences. 

I would add a hard terminal-pair rule:

```text
If a pair contains a single-cell immediate win:
  evaluate/search it as a terminal-equivalence class;
  do not let canonical row order decide whether filler is physically applied;
  do not train pair-level filler preference;
  replay selected_pair but mark terminal_equivalence_class.
```

For actual engine application, either apply only the winning cell and end the game, or apply a documented deterministic order but explicitly exclude filler history from training. The current description does not make this safe enough.

---

## Major issue 4: the two-pass model path may be too expensive

For every expanded full-turn node, Python does:

```text
proposal graph without pair rows -> model forward
candidate selection
graph with admitted pair rows -> model forward
```

The description says both root and interior nodes use this same two-stage pipeline. 

That is architecturally clean, but it may be very expensive. The V1 proposal’s performance contract says interior lifecycle should be one neural evaluation, one candidate-reservoir build, and one bounded scoring pass per expanded full-turn node; widening must not call the model once per pair or once per widening event. 

The implementation can still satisfy the spirit of that contract if the second pass is cheap and reuses the trunk/LEGAL embeddings. But if the second pass recomputes the graph encoder, then every expanded node costs two trunk evaluations. That could make pair-action MCTS lose the equal-wall-clock comparison even if the search semantics are better.

I would strongly prefer:

```text
one encoder pass:
  state/legal/window embeddings
  value
  cell proposal outputs
  legal proposal projections

candidate selection in Python

cheap pair scoring:
  pair_joint_logits = pair scorer over cached legal embeddings + pair features
```

That can be implemented as a second lightweight module call, but not as a full graph re-encode. At minimum, telemetry should separately report:

```text
encoder_forward_count_per_expanded_node
pair_scorer_forward_count_per_expanded_node
proposal_forward_latency
pair_scoring_latency
total_gpu_time_per_expansion
```

“Model eval count” alone is too coarse.

---

## Major issue 5: expansion batching is underspecified

The loop is described as:

```text
requests = v1_engine.run_search_step(max_expansions)
for request in requests:
    complete that expansion with model output
```

That reads as though Python may complete expansions one at a time. If so, GPU utilization will be poor, especially because each expansion may require a proposal pass and a final pair-scoring pass. The proposal’s acceptance gates explicitly include batching, queue backpressure, GPU utilization, candidate-generation p95 latency, neural calls per expanded full-turn node, and pair-scores/sec. 

I would require the provider to batch at two levels:

```text
Batch all proposal graphs for expansion requests.
Run one proposal batch.
Build candidates for all returned nodes.
Batch all admitted-pair scoring graphs.
Run one pair-scoring batch.
Complete all expansions back into Rust.
```

The implementation description says the provider “builds graph batches,” but the lifecycle section describes per-request completion.  I would make the batched path explicit and test it, because this architecture is likely wall-clock constrained rather than simulation-count constrained.

---

## Major issue 6: legal-row hash is not enough; add full state identity

Rust sends Python compact move history bytes, legal row table, tactical payload, and legal row table hash. Python builds the model graph from move history and validates legal-row identity. Rust validates the legal row hash when completing expansion. 

That is good, but I would add a **full Rust state hash**:

```text
state_hash = hash(
  current_player,
  stones by player,
  move count,
  terminal status,
  legal row table identity,
  tactical payload version
)
```

Why? Because two different board positions can plausibly have the same legal-row set or same legal-row count. The model graph is built from move history, not directly from Rust’s in-memory node state. If Python reconstructs the wrong board but the legal rows happen to align, the current validation might not catch it.

I would send:

```text
rust_state_hash
rust_stone_hash
current_player
move_count
legal_row_table_hash
tactical_payload_hash
```

and require Python’s graph builder to echo the same values back in expansion completion metadata. Legal-row identity is necessary but not sufficient.

---

## Major issue 7: progressive widening priors need dynamic normalization

The implementation caches a full scored reservoir, then reveals more rows as:

```text
revealed_limit = ceil(c_pw * max(1, node_visits) ** alpha_pw)
```

PUCT selection only considers revealed rows. 

The question is: are priors normalized over the **full reservoir** or only the **currently revealed subset**?

If priors are softmaxed over the full reservoir but PUCT only considers revealed rows, then the revealed prior mass may sum to much less than 1. That suppresses exploration early and makes the PUCT constant depend on hidden unrevealed candidates. If priors are renormalized over revealed rows, then PUCT behaves more naturally over the current action set.

I would store raw prior logits and compute:

```text
prior_revealed(a) =
  softmax(logit(a) / temperature over revealed selectable candidates)
```

at selection time.

When widening reveals new rows, the revealed prior distribution changes. That is fine, but it should be deliberate and logged. Progressive widening is a known way to make MCTS usable in large or continuous action spaces, but it changes the statistical meaning of visit counts and action priors. ([AAAI Publications][2])

---

## Major issue 8: proposal correction is not safe unless mode semantics are enforced

The implementation admits pairs with:

```text
pair_logits
correction_weights
correction_modes
```

Rust validates that weights are finite and stores corrected prior logits. 

That is not enough.

The V1 proposal correctly distinguishes stochastic samples, deterministic top-k, tactical-protected candidates, structured quota candidates, diagnostic canaries, and unknown inclusions. It also says deterministic top-k, tactical-protected, and quota candidates must not be treated as if they have clean stochastic inclusion probabilities. 

So Rust should enforce mode-specific rules:

```text
exact_importance:
  require exact inclusion probability;
  require 0 < q <= 1;
  allow correction.

clipped_propensity:
  require clip bounds;
  log clipping;
  allow search but mark target caveat.

uncorrected_logged:
  no correction in search;
  eligible for ordinary candidate-conditional target only.

training_forbidden:
  may be used for diagnostics/search depending on config;
  excluded from policy/ranking loss.
```

I would also separate **search priors** from **training corrections**. For PUCT, I would default to:

```text
prior = softmax(model_pair_joint_logits over selectable candidates)
```

and use proposal correction only in specifically corrected training estimators. Otherwise a rare diagnostic canary can get an artificially inflated prior merely because its proposal probability was low.

---

## Major issue 9: the root target still risks being visit-count biased

The implementation stores root visit counts, allocations, Q values, completed Q values, and an improved policy target. 

That is promising, but the document does not define how the improved policy target is built. That is not a minor omission. In this architecture, raw visits are biased by:

```text
candidate admission
protected tactical candidates
Gumbel/root admission behavior
progressive widening
revealed-count timing
source quotas
canaries
late expansion requests
```

Sampled MuZero is relevant here because it explicitly frames planning and learning over sampled action subsets, not over a fully enumerated action space. It argues that only small subsets of actions may be sampled for policy evaluation and improvement in complex action spaces. ([arXiv][3])

The pair target should therefore be explicitly candidate-conditional:

```text
target support = admitted candidate set
not full legal pair table
```

And for low-visit / widened roots, I would use a completed-Q or regularized posterior target rather than raw normalized visits:

```text
π*(a | C) ∝ exp((prior_logit(a) + β * completed_Q(a)) / τ)
```

This aligns better with the regularized policy-optimization view of MCTS, where AlphaZero-style search heuristics can be interpreted as approximations to a regularized policy-optimization problem. ([Proceedings of Machine Learning Research][4])

---

## Major issue 10: final action selection may overweight early visits

The implementation says final root selection orders by:

```text
1. visit count
2. completed Q
3. prior
4. deterministic pair-key tie break
```

That is conventional AlphaZero-like behavior, but this is not conventional AlphaZero. The action set is sampled, root admission may be Gumbel-driven, and progressive/widening-style effects can make visit count reflect **opportunity to be searched**, not only action quality. 

I would make final selection configurable by phase:

```text
training/self-play:
  sample or choose from improved policy target, temperature-controlled

evaluation:
  choose argmax completed_Q among sufficiently visited/admitted candidates

Gumbel-root mode:
  choose the action selected by the Gumbel sequential-halving procedure
```

If final move choice is always “most visits first,” then the system may be quietly optimizing for early admission and PUCT allocation rather than best pair quality.

---

## Major issue 11: the `one_placement` phase is dangerous unless very narrowly defined

The implementation says root phases include:

```text
opening_single
one_placement
terminal
normal_two_placement
```

and for `one_placement`, “Rust selects a terminal single if available.” 

This must not be triggered merely because a 5-window exists. In Hexo, after the opening, every turn consists of two placements. A 5-window is an immediate win because one placement wins and the other placement is filler, but it is still a normal two-placement turn under the game rules. 

The architecture explicitly says hardcoded non-search actions are allowed only for true one-placement exceptions, while normal two-placement hot-window states are searched and tactical facts are proposals/labels, not forced moves. 

I would rename the phase to something harder to misuse:

```text
opening_single_exception
engine_declared_single_placement_terminal_exception
normal_two_placement
terminal
```

And add a test:

```text
Given a normal post-opening state with a 5-window immediate win:
  phase must be normal_two_placement;
  pair search must run;
  tactical winning single may protect terminal-equivalent candidate pairs;
  Rust must not force a single placement.
```

This is a high-severity contract test.

---

## Major issue 12: start-of-turn pair legality still needs a hard proof

The V1 pair contract says both cells reference the same original start-of-turn legal table and the first placement does not expand, shrink, or reorder the second-placement table. 

The game guide says a stone may be placed on an empty hex within distance 8 of any existing stone. 

Those are compatible only if the engine defines two-placement turn legality using the start-of-turn legal set, or if the implementation deliberately restricts the game to that interpretation. If “existing stone” includes the first stone placed earlier in the same turn, then a second placement could become legal only after the first placement. In that case, unordered start-of-turn pairs would exclude legal ordered moves.

I would add a Rust property test:

```text
For every normal two-placement state:
  legal_second_after_first == start_legal_set - {first_cell}
```

If that test fails under the actual game rules, then unordered pair-action MCTS is not implementing the real game.

---

## Major issue 13: per-candidate metadata may be getting compressed into aggregate telemetry

The architecture requires every admitted candidate to carry source contributions, proposal propensity metadata, forced flags, terminal flags, target-support flags, admission generation, and root/interior identity. 

The implementation description says telemetry includes candidate pairs, selected pair, root visits, Q values, completed Q values, support flags, Gumbel/admission order, and selector metrics such as source/eviction/canary/protected counts. 

Counts are not enough. The training target builder needs per-candidate provenance. For each candidate row, replay should preserve something like:

```text
candidate_id
pair_key
source_contributions[]
inclusion_kind
correction_mode
exact_inclusion_probability or heuristic_propensity
forced_exploration_flag
terminal_exact_flag
terminal_equivalence_flag
target_support_flags
root_or_interior
```

If the implementation only stores aggregate source counts, then later training cannot distinguish:

```text
searched and weak
unsampled
diagnostic canary
tactical-protected
deterministic top-k
explicit sampled negative
```

That would break the “unsampled legal pair != negative pair” guarantee in practice, even if the code says it conceptually.

---

## Major issue 14: value backup deserves a dedicated sign-convention test suite

The value convention is:

```text
Python/model returns value from current-player perspective at expanded node.
Rust converts it to root-player perspective.
During backup, Rust updates each edge/node in the appropriate local perspective.
```

That is reasonable. But pair-action games have a few easy-to-miss cases:

```text
terminal after first placement
terminal after second placement
leaf value at opponent-to-move node
backup through root edge only
backup through root + multiple interior nodes
pending expansion path after reused child node
```

The implementation should include explicit tests where the expected value signs are known by construction:

```text
root player immediate pair win -> root edge Q = +1
root player pair allows opponent immediate win -> child value conversion negative
opponent interior pair win -> root perspective negative
terminal after first stone in pair -> same value as terminal after second stone
```

I would not trust this until those tests exist. A sign bug can produce superficially plausible training curves while teaching the policy the wrong player’s preferences.

---

## Major issue 15: parent-key-derived node keys intentionally avoid transpositions

The implementation derives a child node key from the parent node key plus pair key. 

That is simple and probably acceptable for V1. But it means the recursive search is a tree, not a graph. In a placement-only game, the same board state can sometimes be reached by different histories or different pair groupings, assuming no earlier terminal condition. A state-hash transposition table could reuse evaluations and edge statistics more aggressively.

I would not block V1 on this. But I would log transposition opportunities with a Rust board hash:

```text
same board_hash reached by multiple node_keys
```

If this happens often, a future “pair MCTS graph” version could gain a lot. If it almost never happens in practice, the simpler parent-key tree is fine.

---

## Smaller but important improvements

The implementation should define whether blind canaries are allowed to be selected as final moves. The selector says blind canaries are diagnostic/training-forbidden where appropriate, and Rust excludes training-forbidden rows from Gumbel admission in one path.  That is ambiguous. Either canaries are real searchable legal actions that can win if search proves them good, or they are diagnostics that must never be final actions. Both are defensible; mixing the two is not.

The implementation should store raw logits in addition to corrected logits and priors. Debugging proposal correction, prior calibration, and target construction will be much easier if replay has:

```text
model_pair_joint_logit
correction_weight
correction_mode
corrected_prior_logit
prior_after_temperature
prior_over_revealed_subset
```

The implementation should explicitly test D6 identity through the full Rust/Python/Rust loop. The game guide notes that hex grids have D6 symmetry and that axes permute under rotation; the proposal requires D6-stable legal-row and pair-row identity.   Shape tests are not enough; you need transformed search targets and selected pairs to map back correctly.

The implementation should add a tiny-state exhaustive audit mode. On states with small legal-cell count, enumerate all legal pairs, run exhaustive scoring/search, and compare:

```text
candidate reservoir recall
pair prior ranking
candidate-conditional target
full legal-pair target
marginal projection
conditional projection
```

This is the fastest way to catch support bias.

---

## What the research precedent suggests

The implementation is closest to **sampled-action neural planning**, not vanilla AlphaZero. Sampled MuZero is the most relevant precedent because it explicitly addresses settings where full action enumeration is infeasible and policy evaluation/improvement must happen over sampled action subsets. That supports the general idea of bounded pair reservoirs, but it also implies that the target must be candidate-conditional and the candidate sampler is part of the learning algorithm, not just an optimization detail. ([arXiv][3])

The Gumbel literature supports the desire to avoid raw visit-count AlphaZero targets when not all root actions are visited. But it does not support merely adding a Gumbel tie-break to PUCT and calling that Gumbel policy improvement. The paper’s key point is policy improvement through sampled actions without replacement, and better behavior with few simulations. ([OpenReview][1])

Progressive widening is a reasonable tool for large action spaces, including continuous-action MCTS variants, but it changes the interpretation of visits because actions are not all available from the beginning. That reinforces the need for completed-Q or regularized posterior targets rather than naive root visit normalization. ([AAAI Publications][2])

Connect6 is also a useful warning. It is a two-stone-turn game with enormous branching and sudden-death tactical structure; published Connect6 MCTS work used threat-space search plus MCTS because ordinary MCTS struggles with the branch/threat structure. That supports the tactical-candidate side of this implementation, but it also suggests that search traces and tactical fixtures are mandatory, not optional. ([ResearchGate][5])

---

## Acceptance criteria I would add before trusting this

I would require these before considering the implementation credible:

```text
1. Root Gumbel trace:
   show considered set, Gumbel values, sequential-halving allocation,
   completed-Q, improved target, selected action.

2. PUCT first-selection test:
   prove priors affect first edge selection or define intentional tie behavior.

3. Terminal-equivalence test:
   pair {winning_cell, filler_cell} does not train filler preference
   and does not produce order-dependent replay semantics.

4. Start-of-turn legality property test:
   second-placement legality cannot expand due to first placement.

5. State identity roundtrip:
   Rust state hash -> Python graph -> Rust completion hash.

6. Revealed-prior normalization test:
   PUCT priors over revealed rows behave as intended after widening.

7. Per-candidate metadata replay test:
   source contributions and correction modes survive candidate selection,
   Rust admission, search, final replay, and target construction.

8. Backup sign tests:
   terminal first placement, terminal second placement, opponent leaf,
   multi-depth path.

9. Batched expansion performance test:
   expansion requests are actually batched across proposal and pair-scoring passes.

10. Equal-wall-clock baseline:
   compare against autoregressive/sequential afterstate-DAG MCTS,
   not just an older weak pair-prior mode.
```

The implementation document itself says Rust compile/tests and PyO3 lifecycle tests still need to run in a Rust-capable environment, and final V1 acceptance still requires search traces, performance profiles, recall artifacts, D6 artifacts, and equal-wall-clock arena scorecards. That boundary is appropriate; this should not be considered accepted until those artifacts exist. 

---

## Final opinion

This implementation is directionally strong, but I would call it **V1-pair-search-shaped**, not yet proven V1 pair search.

The biggest correction is conceptual: **do not let “candidate reservoir + PUCT + Gumbel tie-break” masquerade as “Gumbel sequential-halving pair MCTS.”** Either implement the real root Gumbel policy-improvement procedure, or honestly label this as sampled pair PUCT with progressive widening.

The second biggest correction is statistical: **the replay target must be a candidate-conditional completed-Q / regularized target**, not raw root visits. Raw visits are too entangled with candidate admission and widening.

The third biggest correction is operational: **make the two-pass model path cheap and batched**, or the architecture may lose on wall-clock even if the search is strategically better.

With those fixes, this is a plausible and testable implementation. Without them, the main risk is that you spend a lot of engineering effort testing a distorted version of the original hypothesis: not “does pair-action MCTS work?”, but “does a costly sampled-pair PUCT variant with biased targets and unclear Gumbel semantics work?” Those are very different experiments.

[1]: https://openreview.net/forum?id=bERaNdoegnO&utm_source=chatgpt.com "Policy improvement by planning with Gumbel"
[2]: https://ojs.aaai.org/index.php/AAAI/article/view/5885/5741?utm_source=chatgpt.com "Monte-Carlo Tree Search in Continuous Action Spaces ..."
[3]: https://arxiv.org/abs/2104.06303?utm_source=chatgpt.com "Learning and Planning in Complex Action Spaces"
[4]: https://proceedings.mlr.press/v119/grill20a.html?utm_source=chatgpt.com "Monte-Carlo Tree Search as Regularized Policy Optimization"
[5]: https://www.researchgate.net/publication/220437214_Two-Stage_Monte_Carlo_Tree_Search_for_Connect6?utm_source=chatgpt.com "Two-Stage Monte Carlo Tree Search for Connect6"
