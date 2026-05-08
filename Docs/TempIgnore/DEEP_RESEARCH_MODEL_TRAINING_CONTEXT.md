# Deep Research Context: Hexo Model Architecture And Training

This document is intended as a handoff to a frontier deep-research model. The
goal is not to summarize implementation details exhaustively, but to explain the
game, the current architecture direction, the training problem, and the weakest
assumptions that need state-of-the-art research review.

## Research Assignment

Please evaluate the best model architecture and training strategy for Hexo-RL,
with special focus on first-class joint pair MCTS. We need a research-level
hypothesis for how to train a superhuman model for this game. The strongest
current hypothesis is that a complete two-placement turn should be treated as
one search action `(q1, r1, q2, r2)`, not as two separate placement decisions.
Please challenge that assumption, compare it against alternatives, and identify
the safest SOTA path.

Research should cover:

```text
1. Search architecture:
   How should MCTS represent two-placement turns?
   Should pair actions be first-class macro-actions?
   Should search use progressive widening, sampled actions, factored actions,
   Gumbel candidate admission, or another SOTA method?

2. Model architecture:
   What network design should score legal single actions and legal pair actions?
   Is a graph/set/attention model the right primary architecture?
   Should pair scoring be biaffine, pointer-style, autoregressive,
   set-transformer-like, energy-based, or something else?

3. Training:
   Can sparse joint pair visit targets train well?
   How should early training avoid collapse when the policy is random?
   What bootstrapping, curriculum, auxiliary losses, replay balancing,
   calibration, and evaluation are needed?

4. Game-specific strategy:
   How should the model learn threat overload, pair blocking, 4-window and
   5-window equivalence, multi-axis pressure, and unbounded-board local/global
   balance?
```

## Game Context

Hexo is similar to Connect6, but played on an infinite hexagonal grid using
axial coordinates `(q, r)`. It is not the game Hex. Player 0 opens by placing
one stone at `(0, 0)`. After that, players alternate turns, and each turn places
two stones by the same player. Player 1 gets the first two-stone response.

Legal placements are empty hexes within hex distance 8 of any existing stone.
Hex distance uses cube distance:

```text
distance = max(|dq|, |dr|, |dq + dr|)
```

The win condition is six stones in a row along any of the three hex axes:

```text
(1, 0)
(0, 1)
(1, -1)
```

The board has no fixed edges or corners. There is no draw in normal theory
because the board is infinite, though training games can still be truncated by
engineering limits.

## Strategic Properties

The two-placement turn structure is the central strategic fact. A 5-window
containing five stones and one empty cell wins with one placement. A 4-window
containing four stones and two empty cells also wins in one turn, because the
player can fill both empty cells. This makes 4-windows much closer to immediate
threats than in Gomoku-like one-placement games.

The core win mechanism is threat overload. A single hot window can usually be
blocked. The winning player tries to create more independent hot windows than
the opponent's two placements can cover. The key strategic objects are:

```text
six-cell windows
4-window and 5-window hot threats
block-cell sets
pairs of cells that cover or fail to cover multiple threats
multi-axis pressure
shared pivot cells
forks and threat overload
```

This is why pair-level reasoning may matter so much. Many important decisions
are not "which single cell is best?" but "which two cells together create,
complete, or cover the right set of windows?"

Hex grids have D6 symmetry: 6 rotations and 6 reflections. Spatial transforms
must also permute axis-related labels and pair rows consistently.

## Current System Shape

Hexo-RL is an AlphaZero-style single-machine training system:

```text
Rust rules and MCTS engine
Python self-play workers
Python shared-memory inference server
Python replay buffer and sampler
Python PyTorch trainer
dashboard and metrics
```

Rust owns rules, legal move generation, placement phase rules, win detection,
MCTS, and legality validation. Python owns model definitions, inference
batching, replay, training, and orchestration.

The current training loop is:

```text
self-play / bootstrap records
-> replay target processing
-> ReplayDataset
-> model forward
-> policy/value/auxiliary losses
-> optimizer step
-> EMA checkpoint
-> evaluation / arena
```

Current model families include:

```text
dense CNN / residual model over a 33x33 crop
sparse candidate policy heads
crop-compatible pair policy head
global graph model
global graph pair heads
auxiliary value, tactical, axis, regret, lookahead, and opponent-policy heads
```

The current docs increasingly favor global graph models because the game is
unbounded and legal actions are row-table objects, not fixed board pixels.

## Current Global Graph Direction

The global graph model sees a variable-length list of game objects:

```text
STATE token
TURN token
STONE tokens
LEGAL action tokens
WINDOW6 tokens
relations between tokens
```

The model outputs policy logits over legal action rows. This is different from
the CNN path, which outputs dense logits over a fixed 33x33 board crop. The
graph path better matches an unbounded game because a legal action row can be
identified by a global coordinate rather than a crop index.

The minimal graph-token plan intentionally removes expensive or redundant token
types from the hot path. Pair actions should not become attention tokens by
default, because there can be `O(A^2)` legal pairs for `A` legal actions. The
preferred design is to score pair rows by referencing existing legal token
vectors rather than materializing every pair as a graph token.

Current pair heads are:

```text
policy_pair_first:
  one logit per legal first-placement row

policy_pair_joint:
  one logit per canonical unordered first-placement pair row

policy_pair_second:
  one logit per known-first second-placement row
```

Current `policy_pair_joint` uses symmetric pair features:

```text
state
first_vector + second_vector
abs(first_vector - second_vector)
first_vector * second_vector
```

A suspected stronger design is a full legal-row interaction scorer, for example
biaffine or pointer-style scoring:

```text
score(i, j) = h_i^T U h_j + w^T [h_i, h_j, state]
```

This should be investigated carefully. The best architecture may need to score
all or many legal pairs efficiently without putting pair rows into the attention
sequence.

## Current Pair-MCTS Hypothesis

The current strongest hypothesis is described in
`Docs/JOINT_PAIR_MCTS_RESEARCH_NOTE.md`: represent a two-placement turn as one
first-class pair action. MCTS children are pair actions. Expansion applies both
stones atomically. The policy target is the MCTS visit distribution over joint
pairs.

Because the pair action space is quadratic, the current hypothesis does not
recommend full enumeration everywhere. It recommends:

```text
sampled candidate admission
Gumbel sampling without replacement
progressive widening
nonzero uniform exploration floor
tactical pair proposals
factored proposal helpers
sparse joint pair replay targets
```

The proposal distribution might look like:

```text
beta(a | s) =
  lambda_policy   * joint_pair_policy(a | s)
+ lambda_uniform  * uniform_legal_pairs(a | s)
+ lambda_tactical * tactical_pair_sampler(a | s)
+ lambda_factored * factored_pair_sampler(a | s)
```

The uniform term should anneal but not vanish, because deterministic top-k from
an immature pair policy can permanently hide important pairs from search.

## Training Targets Under Consideration

The proposed primary policy target is sparse pair visit mass:

```text
pi_pair(a) = N(s, a)^(1 / tau) / sum_b N(s, b)^(1 / tau)
```

Stored training data would include:

```text
canonical pair rows
visit counts
candidate set
proposal source per candidate
proposal probability or weight
total legal pair count
legal-pair mask or row-table hash
D6 transform metadata
root value and child Q summaries
game outcome
```

Auxiliary single-placement targets can be derived by marginalizing:

```text
pi_cell(i) = sum_j pi_pair(i, j)
```

but the current hypothesis says these should be auxiliary only. They should not
define the main policy if pair synergy is the strategic object.

## Known Technical And Conceptual Weak Points

These are the areas most in need of deep research review.

### 1. Pair-As-Action May Be Correct But Expensive

Treating the pair as one macro-action preserves strategic semantics, but the
branching factor is enormous. Progressive widening and sampled actions are
plausible, but we need stronger evidence that this will outperform a sequential
or autoregressive search at equal wall-clock.

Research question:

```text
Should Hexo search use first-class pair actions, sequential afterstates,
autoregressive pair choice, factored-action MCTS, or a hybrid?
```

### 2. Candidate Admission May Bias Training

If only sampled pair candidates receive visits, the target distribution is
partly shaped by the proposal distribution. This can create a self-confirming
loop: the model proposes pairs, MCTS only searches those pairs, and training
reinforces them.

Research question:

```text
What correction or logging is needed when training from sampled-action MCTS?
Should targets include importance correction, proposal-aware loss terms,
unvisited legal-pair negatives, or only sparse positives?
```

### 3. Early Training Could Collapse

Early pair policies will be random or poorly calibrated. Gumbel is useful for
sampling without replacement, but it is not magic; if the proposal is dominated
by a bad policy, search can miss good pairs. The current answer is a uniform
floor, tactical proposals, and high entropy. We need better SOTA guidance.

Research question:

```text
How should exploration be scheduled from random policy to mature policy in a
quadratic action space?
```

### 4. The Best Pair Scorer Is Unknown

Current pair scoring is an MLP over pair features. Biaffine, pointer-network,
set-transformer, cross-attention, energy-based, and autoregressive models may
all be better. The model must score many candidate pairs efficiently and
preserve D6 symmetry or equivariance as much as possible.

Research question:

```text
What architecture best scores pair interactions over a legal-row set without
materializing O(A^2) attention tokens?
```

### 5. Tactical Priors Could Help Or Poison Learning

Hexo has strong tactical structure: 4/5-windows, block sets, cover pairs, and
threat overload. Injecting tactical proposals may help exploration, especially
early. But overly strong hand-coded tactical channels could make the model
depend on brittle heuristics or duplicate what MCTS should discover.

Research question:

```text
Which tactical objects should be model inputs, proposal sources, auxiliary
targets, or purely evaluation diagnostics?
```

### 6. Value Learning May Be Higher Variance

A pair action commits two stones at once. The value target says whether the
resulting game was won, but credit assignment inside the pair is blurry. Bad
early pair actions may produce unstable value targets and search-confirmation
bias.

Research question:

```text
What value targets, auxiliary Q/regret heads, or bootstrapping methods reduce
variance for pair-level MCTS?
```

### 7. Replay Distribution May Be Skewed

Openings have huge legal-pair counts. Late tactical states may have smaller
pair spaces but higher strategic importance. Uniform replay over positions may
undertrain rare tactical pair decisions.

Research question:

```text
How should replay be balanced by game phase, legal-pair count, entropy,
regret, tactical density, and search surprise?
```

### 8. Evaluation Must Be Wall-Clock Aware

Pair-level search may look strong at fixed simulation count while losing at
fixed wall-clock because pair candidate scoring is expensive. The research plan
must define evaluation that distinguishes algorithmic strength from compute
overhead.

Research question:

```text
What experiments prove pair-level MCTS is superior at equal time, equal
training budget, and equal inference budget?
```

### 9. Curriculum Could Mislead

Small boards or reduced legal sets may make pair MCTS trainable early, but the
game is infinite-board and radius-limited. A curriculum that changes the action
structure too much might not transfer.

Research question:

```text
What curriculum, if any, preserves the strategic distribution of full Hexo?
```

### 10. Symmetry And Row Identity Are Easy To Get Wrong

D6 transforms must correctly transform single actions, pair rows, windows,
axes, and policy targets. Since pair rows are unordered in first-placement
states, canonicalization must be consistent before and after augmentation.

Research question:

```text
What row identity, hashing, augmentation, and target-validation scheme is
needed for reliable pair training?
```

## Candidate Architectures To Compare

Please compare at least these research families:

```text
1. Pair-action MCTS with sampled/Gumbel/progressive widening.
2. Sequential MCTS with first-placement node and second-placement afterstate.
3. Autoregressive pair policy: p(first | s) * p(second | s, first).
4. Factorized-action MCTS with internal cell-factor statistics but pair-edge
   backups.
5. Full or sampled legal-pair matrix scorer with biaffine/pointer scoring.
6. Energy-based pair scorer over legal-row sets.
7. Pure graph model with WINDOW6 tokens and pair scorer.
8. Graph plus explicit tactical proposal/evaluation module.
9. MuZero-style learned dynamics over pair macro-actions.
10. Regret-guided or search-control augmented self-play for hard pair states.
```

## Desired Output From Deep Research

Please produce a research report with:

```text
recommended architecture
recommended search algorithm
recommended exploration schedule
recommended pair scorer design
recommended training targets and replay format
recommended bootstrap/curriculum, if any
key risks and mitigations
experiments/ablations to run first
success metrics and failure criteria
SOTA papers and why they apply
what current assumptions should be rejected or downgraded
```

The most useful output would be a decisive plan for the best possible approach,
not a generic survey. Implementation cost can be assumed free for research
purposes, but the final plan should still distinguish fixed-simulation strength
from fixed-wall-clock strength.

## Local Reference Documents

Use these project docs as context:

```text
Docs/game.md
Docs/SYSTEM_DESIGN.md
Docs/TRAINING.md
Docs/GLOBAL_GRAPH_TOKEN_INPUTS.md
Docs/GLOBAL_GRAPH_MINIMAL_TOKEN_PLAN.md
Docs/GLOBAL_GRAPH_PAIR_HEADS.md
Docs/JOINT_PAIR_MCTS_RESEARCH_NOTE.md
Docs/OPTUNA_SEQUENTIAL_AUTOTUNING_PLAN.md
Docs/RGSC_IMPLEMENTATION.md
```

Important caveat: some docs describe current code and some describe planned
directions. Treat this document as the current research handoff and validate the
architecture choices independently.
