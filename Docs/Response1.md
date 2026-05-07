# Hexo-RL research report: recommended path

My decisive recommendation is **not** “plain joint-pair MCTS.” It is:

**Use first-class unordered pair actions as the actual MCTS edges, but only through a sampled, proposal-aware, Gumbel/progressive-widened candidate system with exact tactical admission.** The model should be a **global graph/set encoder over stones, legal cells, and WINDOW6 objects**, with a **symmetric low-rank biaffine pair scorer plus tactical pair features**, and with **single-cell and autoregressive heads only as auxiliaries/proposal helpers**.

That is the strongest architecture for fixed simulation count. For fixed wall-clock, it must be implemented as a two-stage fast path: cheap single/cell/tactical proposal first, then pair scoring only on a bounded candidate set. The sequential-afterstate approach is valuable as a baseline and proposal generator, but I would not make it the main search target.

The project context correctly identifies the core reason: in Hexo, after the opening each turn places two stones; 4-windows and 5-windows are both win-in-one-turn objects; and the key decisions are often pair decisions that create, complete, or cover multiple windows, not independent single-cell choices. 

---

## 1. Recommended search architecture

### Verdict

Use **Sampled-Gumbel Pair AlphaZero**:

1. **MCTS child = one unordered canonical pair** `(cell_i, cell_j)`.
2. **Transition applies both stones atomically** from the turn state.
3. **Candidate set is sampled, not fully enumerated**, except in small/tactical states.
4. **Every expanded node stores proposal probability/inclusion metadata** for each admitted pair.
5. **PUCT/Gumbel search uses proposal-aware priors**, not raw model logits.
6. **Exact tactical logic force-admits immediate wins, immediate blocks, and threat-overload cover pairs.**
7. **Progressive widening adds more pair candidates as node visits grow.**

This is the right compromise between semantic correctness and quadratic action cost. AlphaZero’s general policy/value/MCTS loop is the right outer framework: it trains a network from self-play, using search visit distributions as policy targets and outcomes as value targets; AlphaGo Zero describes the MCTS edge statistics, count-based root policy target, and policy/value training pattern directly relevant here.  Sampled MuZero is the directly applicable SOTA idea for enormous structured action spaces: it explicitly addresses cases where full action enumeration is infeasible and uses sampled action subsets for policy improvement/evaluation. 

### Why pair macro-actions should be first-class

A Hexo “move” is not “first stone, then second stone” strategically. It is a **set of two placements**. Many decisive actions have low single-cell marginal probability but high pair value: two cells may jointly complete a 4-window, jointly block two hot opponent windows, or create overload across axes. A sequential search has to invent an artificial afterstate after the first stone where the same player still moves. That afterstate is not an opponent-facing game state, so its value target is structurally artificial.

At fixed simulation count, pair MCTS is stronger because each simulation evaluates a real game decision. Sequential MCTS spends part of the budget resolving internal ordering and duplicate pair representations. It also makes the first-cell target vulnerable to marginalization bias: a cell that is only good with one specific partner may look bad as a first action.

### Why plain pair MCTS is still wrong

Full pair enumeration at every node should be rejected. If there are `A` legal cells, there are `A(A-1)/2` unordered pairs. Even if the model can score the root matrix, doing that at every node destroys wall-clock efficiency. This is exactly the regime where Sampled MuZero-style action subsets and progressive widening apply: sample a small subset, plan over it, and log enough information to account for the sampling process. Sampled MuZero notes that full enumeration can be infeasible and that sampling distribution/procedure must be understood for policy improvement over the whole action space, not just the sampled set. 

### Concrete node expansion policy

At each node:

**Step A: exact tactical admission.** Before sampling, admit all pairs in these categories:

* Immediate winning pairs: fill a 4-window’s two empties; fill a 5-window’s empty plus any legal filler.
* Immediate forced blocks: pairs that hit every opponent hot window when the opponent has win-next-turn threats.
* Minimal cover pairs for opponent threat overload.
* High-value tactical creation pairs: pairs creating multiple independent 4/5 windows, shared-pivot forks, or multi-axis pressure.

For 5-window wins, I recommend a **terminal single-completion action class** internally. If one cell wins immediately and the second stone is irrelevant to the game result, do not let MCTS spread target mass over thousands of equivalent `(winning_cell, filler)` pairs. Represent it as a special terminal class during search/training, then choose any legal filler only for rules logging. This will substantially reduce target fragmentation.

**Step B: sample non-tactical candidates from a mixture proposal**:

[
\beta(a|s)=
\lambda_p,p_\theta^{pair}(a|s)
+\lambda_u,U(a|s)
+\lambda_t,T(a|s)
+\lambda_f,F(a|s)
]

where `T` is tactical proposal, `F` is factored/single-cell proposal, and `U` is true or stratified uniform pair sampling. The uniform component should shrink but never reach zero.

**Step C: Gumbel top-m admission.** Use Gumbel top-k without replacement from `β` to admit diverse candidates. Gumbel AlphaZero/Gumbel MuZero is relevant because it was designed for the failure mode where AlphaZero does not visit all root actions; it samples actions without replacement and uses sequential halving to improve planning with few simulations. ([OpenReview][1]) The paper’s algorithm samples Gumbels, takes top actions by `g + logits`, and uses sequential halving to allocate simulations among them. 

**Step D: proposal-aware PUCT.** Do not use raw pair logits as if the candidate set were unbiased. Use a clipped normalized correction such as:

[
P_C(a|s) \propto \exp(z_\theta(a|s)-\log \beta(a|s))
]

over the admitted candidate set, with clipping on `-log β` to prevent extreme weights. For Gumbel-without-replacement, use inclusion probability if you estimate it; otherwise log the sampling source and use `β` as the practical correction. Sampled MuZero explicitly warns that naive sampled-action PUCT can be unstable and proposes using a modified prior so the visit counts already account for the sampling procedure. 

**Step E: progressive widening.** Start each node with `K0` candidates, then admit more when:

[
|C(s)| < K_0 + c,N(s)^\alpha
]

with `α ≈ 0.35–0.6`. Use larger `K` at roots and in high-entropy/high-tactical-density states. Progressive widening exists for exactly this purpose: controlling bias/variance when a search cannot expand all actions early. Couëtoux et al. describe double progressive widening as a way to balance “infinitely many simulations for each action/state” against having enough nodes to avoid first-node bias. ([Springer Link][2])

### Root selection

Use two root modes:

**Training / low simulation mode:** Gumbel sequential halving over admitted pair candidates. This is best when simulations are scarce. Gumbel MuZero reportedly learned reliably on 9x9 Go with very few simulations where MuZero failed, and its ablations show sensitivity to number of sampled actions. 

**Mature / high simulation mode:** standard PUCT over progressively widened pair candidates, with tactical forced admission. This gives deeper search when the candidate set is already high quality.

---

## 2. Rejected or downgraded search alternatives

| Alternative                                   |                           Decision | Reason                                                                                                      |
| --------------------------------------------- | ---------------------------------: | ----------------------------------------------------------------------------------------------------------- |
| **Full joint-pair MCTS enumeration**          |                             Reject | Correct semantics, impossible branching at most nodes.                                                      |
| **Sampled first-class pair MCTS**             |                          Recommend | Preserves true game action while controlling branching.                                                     |
| **Sequential afterstate MCTS**                |     Downgrade to baseline/proposal | Faster per expansion, but introduces artificial first-stone states, order bias, and marginalization errors. |
| **Autoregressive pair policy only**           |          Use as auxiliary/proposal | Good for generating candidates, too biased as the main policy target.                                       |
| **Factored-action MCTS with cell stats only** |            Use as helper, not main | Pair synergy is the game; pure factorization misses overload and cover-pair structure.                      |
| **Sampled MuZero-style action subsets**       |        Adapt, but with exact rules | The sampled-action idea applies; learned dynamics do not.                                                   |
| **MuZero learned dynamics**                   |             Reject for core system | Hexo has exact deterministic rules; learned dynamics add error without solving the main branching issue.    |
| **Pure tactical/proof search**                | Use for terminal/forced cases only | Hexo needs strategic self-play beyond immediate threats.                                                    |
| **Pure CNN crop model**                       |                          Downgrade | Infinite board and row-table legal actions make graph/legal-token modeling cleaner.                         |

Factored MCTS is worth studying only as a helper. The factored-MCTS literature addresses large factored action spaces, but it also admits the approximations can be heuristic and condition-dependent; pure factorization is dangerous in Hexo because pair interactions are the value. ([AAAI][3])

---

## 3. Recommended model architecture

### Encoder

Use a **global graph/relational set transformer**:

* `STATE` token
* side-to-move / turn-phase token
* stone tokens
* legal-cell tokens
* WINDOW6 tokens
* typed relations: cell-in-window, stone-near-cell, cell-cell axial relation, window-axis relation, ownership/threat relation

This is the right inductive bias because Hexo’s important objects are sparse, relational, and unbounded: stones, legal cells, six-cell windows, block sets, axes, and pairs. Graph networks are relevant because they provide object- and relation-centric inductive bias for systems where interactions matter. ([Proceedings of Machine Learning Research][4]) Set Transformer is relevant because legal cells/windows are variable-size sets and the model should be permutation-invariant or permutation-equivariant over row order; its inducing-point mechanism is also relevant for scalable attention over large sets. ([arXiv][5])

The encoder should not create `O(A²)` pair tokens. It should create legal-cell embeddings `h_i` and window embeddings, then score pair rows by referencing cell embeddings.

### Pair scorer

Use a **symmetric biaffine + feature scorer**, not the current plain symmetric MLP alone.

Recommended form:

[
s(i,j)=
b_i+b_j
+h_i^\top U_\theta(s)h_j
+w^\top \phi(i,j,s)
+\text{MLP}([h_i+h_j,\ |h_i-h_j|,\ h_i\odot h_j,\ r_{ij},\ t_{ij}])
]

where:

* `b_i, b_j` are single-cell priors.
* `Uθ(s)` is a low-rank state-conditioned bilinear/biaffine interaction.
* `r_ij` includes relative axial geometry: distance, same-axis relation, same-window relation, D6-canonical relative vector.
* `t_ij` includes exact tactical pair features: wins now, blocks all hot opponent windows, number of own hot windows created, number of opponent hot windows covered, multi-axis count, shared pivot count, overload score.
* The scorer is symmetric: `s(i,j)=s(j,i)`.

Biaffine scoring is a good fit because it is a proven way to score pairwise arcs over variable token sets; Dozat and Manning used biaffine classifiers to score dependency arcs with strong results. ([arXiv][6]) Pointer Networks are relevant only as an auxiliary/autoregressive proposal mechanism because they handle variable output dictionaries, but they are order-sensitive and should not define the main unordered-pair policy. ([arXiv][7])

### Two-pass scoring for wall-clock

Use two scoring paths:

**Fast coarse path**

1. Score all legal cells.
2. Select top `B` cells plus tactical cells plus random/uniform cells.
3. Form candidate pairs from:

   * top-by-top combinations,
   * tactical pairs,
   * Gumbel samples,
   * factored samples,
   * stratified uniform pairs.
4. Apply pair scorer only to this candidate set.

**Strong path**

At root or analysis mode, compute a coarse low-rank pair matrix for all pairs if `A` is small enough, then refine top candidates. At internal nodes, stay sampled.

### Auxiliary heads

Add these heads, with lower weight than the pair policy:

* Single-cell marginal policy: (\pi_i=\sum_j \pi_{ij})
* Autoregressive first/second policy: (p(i|s)p(j|s,i))
* Immediate win classifier for cells and pairs
* Opponent immediate threat/block classifier
* WINDOW6 type classifier: dead / 3 / 4 / 5 / completed
* Pair coverage classifier: covers all opponent hot windows?
* Pair overload classifier: creates unblockable next-turn threat?
* Child-Q / advantage / regret head over visited candidate pairs
* D6 consistency loss

The single and autoregressive heads are important for gradient sharing and candidate generation. They should not replace the joint pair policy.

---

## 4. Exploration schedule

### Phase 0: tactical bootstrap before serious self-play

Do not start entirely from random pair play. AlphaZero can start from random weights in chess/shogi/Go because the action spaces are manageable enough for MCTS to discover useful moves from broad priors; Hexo’s quadratic pair space makes random pair discovery much worse. AlphaZero’s tabula-rasa self-play result is the right inspiration, but Hexo needs stronger candidate discovery scaffolding. ([Gwern][8])

Create synthetic full-rule positions embedded on the infinite board:

* immediate 5-window wins
* immediate 4-window wins
* one-threat block cases
* two-threat block cases
* impossible-to-block overloads
* multi-axis forks
* shared-pivot windows
* noisy local contexts with irrelevant stones

Train supervised auxiliary heads and a weak pair policy on exact labels. This is not “hand-coded strategy as value”; it is rule-derived bootstrapping for candidate recall.

Connect6 literature supports this emphasis: Connect6 has numerous candidate moves and sudden-death properties, and threat search/threat move generation are central to strong search. ([ScienceDirect][9]) Two-stage MCTS for Connect6 explicitly combined threat-space search with MCTS and found it more efficient than traditional MCTS on positions with threat-space solutions. ([ResearchGate][10])

### Phase 1: early self-play

Use broad exploration:

* `λ_uniform`: 0.40–0.60
* `λ_tactical`: 0.25–0.40
* `λ_policy`: 0.05–0.20
* `λ_factored`: 0.10–0.20

Root candidate count should be large: `K_root ≈ 256–1024` depending on legal count and GPU budget. Internal node `K` can be smaller: `32–128`.

Use:

* Gumbel admission without replacement
* high root temperature
* no resignation
* D6 augmentation
* high entropy regularization
* replay oversampling of tactical states

Do not use deterministic top-k from the model. That is the fastest route to self-confirming collapse.

### Phase 2: mid training

Gradually shift to model-guided search:

* `λ_policy`: 0.40–0.70
* `λ_tactical`: 0.15–0.30
* `λ_factored`: 0.10–0.20
* `λ_uniform`: 0.05–0.15

The true uniform component should not vanish. It can become tiny, but keep it for support and audits.

Use progressive widening more aggressively in high-uncertainty states:

* high policy entropy
* low value confidence
* high tactical density
* large legal pair count
* high disagreement between pair scorer and tactical module
* high root search surprise

### Phase 3: mature training/evaluation

For training, still keep small stochasticity. For evaluation:

* no Dirichlet/Gumbel randomness in final move choice
* larger root `K`
* deterministic best visit / best value action
* optional D6 ensemble in analysis mode
* exact tactical forced resolution before neural search

AlphaGo Zero used visit-count temperature and Dirichlet root noise for training diversity; the same principle applies, but Hexo’s noise should be applied to sampled pair admission/proposal rather than to a fully enumerated action list. 

---

## 5. Can sparse joint pair MCTS targets train well?

Yes, but **not as raw sparse positives over a policy-biased candidate set**.

Sparse pair targets can train if you follow five rules:

### Rule 1: train on candidate-set distributions, not fake full-board negatives

For a search with admitted candidate set `C`, train:

[
\pi_C(a)=\frac{N(a)^{1/\tau}}{\sum_{b\in C}N(b)^{1/\tau}}
]

over `C`. Include admitted-but-unvisited candidates as zero/low mass in the candidate softmax. Do **not** treat every unadmitted legal pair as a negative. Most unadmitted pairs were never evaluated.

### Rule 2: log proposal metadata

Replay records must include:

* canonical pair rows
* candidate set
* visit counts
* proposal source
* `β(a|s)` or inclusion probability
* raw model logits at search time if available
* total legal cell/pair count
* legal row-table hash
* D6 transform metadata
* root value and child-Q summaries
* tactical certificates active at the node

This is mandatory. Without it, you cannot distinguish “bad pair” from “not sampled.”

### Rule 3: use Sampled-MuZero-style correction in search

The cleanest approach is: correct the prior during search, then train the raw model to the corrected visit distribution. Sampled MuZero’s sampled-action framework is relevant because it shows how sampled root actions can be used so that visit counts already account for the sampling procedure, and then projected back to the network by KL minimization. 

### Rule 4: add marginal and autoregressive auxiliaries

A single sparse pair target gives very little gradient to cells that appear in multiple good pairs. Add:

[
\pi_{\text{cell}}(i)=\sum_j \pi_{\text{pair}}(i,j)
]

and an autoregressive auxiliary (p(i|s)p(j|s,i)). These heads stabilize learning, improve candidate recall, and reduce the risk that one good pair receives all credit while equivalent tactical variants are forgotten.

### Rule 5: collapse terminal equivalence classes

If a 5-window completion wins regardless of the second stone, collapse the target into a terminal single-completion class. Otherwise the same winning idea becomes thousands of pair labels, which creates artificial entropy and weakens the policy.

---

## 6. How to avoid collapse and bias

The dangerous loop is:

**model proposes pairs → MCTS only searches those pairs → replay reinforces the same pairs → model narrows further.**

Prevent it with these mechanisms:

1. **Nonzero uniform admission forever.**
2. **Exact tactical admission independent of the neural policy.**
3. **Proposal-aware PUCT.**
4. **Replay reanalysis of hard states with larger/different candidate sets.**
5. **Candidate recall audits:** periodically run near-exhaustive root search on small/legal-local states and compare whether the normal candidate system includes the best tactical pairs.
6. **Entropy floors by game phase:** early openings need much higher entropy than late forced states.
7. **Policy calibration checks:** monitor whether pair logits are calibrated within sampled candidate sets and across proposal sources.
8. **Source-balanced replay:** do not let policy-proposed candidates dominate tactical/uniform discoveries.

The completed-Q policy loss from Gumbel MuZero is also relevant for sparse/low-simulation settings: Gumbel MuZero trains with completed Q-values and reports that without them, related regularized variants failed to produce policy improvement in their setup. 

---

## 7. How the model should learn Hexo-specific strategy

### WINDOW6 tokens should be first-class inputs

Do not expect a generic graph encoder to rediscover all six-cell window semantics efficiently. The model should see WINDOW6 objects with:

* axis
* six cells
* own count
* opponent count
* empty cells
* hot/dead status
* block-cell set
* whether a pair completes it
* whether a pair blocks it
* overlap with other windows
* shared pivot cells

This is not “heuristic poisoning.” These are game-rule-derived objects, like legal moves or attack maps. The value function should still be learned.

### Pair blocking should be a hitting-set feature

For opponent hot windows, each candidate pair should know:

* how many opponent hot windows it hits
* whether it hits all current opponent immediate threats
* whether it blocks one threat but fails another
* whether it blocks while creating own threat
* whether the opponent still has a win after the pair

This is the central pair-level concept. A single-cell policy cannot represent it cleanly.

### Threat overload should be an auxiliary target

Train a binary or ordinal target:

* pair creates 0 hot windows
* pair creates 1 hot window
* pair creates 2 hot windows
* pair creates >2 hot windows
* pair creates no-cover overload

This teaches the network why two individually ordinary cells can be decisive.

### Multi-axis pressure needs relative geometry

The pair scorer should know whether two cells:

* lie on same axis
* participate in different axes through a shared pivot
* complete or extend multiple windows
* create threats with disjoint block sets
* cover threats whose block sets overlap

### D6 symmetry must be enforced, not merely hoped for

Use all 12 D6 transforms in augmentation and property tests. Every transform must correctly map:

* axial coordinates
* axes
* window rows
* legal-cell rows
* unordered pair rows
* candidate-set hashes
* policy target mass
* tactical labels

AlphaGo Zero used dihedral rotations/reflections during neural evaluation in Go; Hexo should use D6 augmentation/consistency for the same reason, but with hex-axis label permutation and unordered pair canonicalization handled explicitly. 

---

## 8. Weakest assumptions in the attached plan

### Weak assumption 1: “Pair-as-action” alone solves pair reasoning

Pair-as-action is semantically right, but **plain pair MCTS is not enough**. The real solution is **pair edges plus sampled/proposal-aware candidate admission plus exact tactics**.

### Weak assumption 2: sparse joint targets are safe by default

They are safe only if proposal metadata is logged and the policy loss is candidate/proposal-aware. Sparse positives without admission correction will amplify the proposal distribution.

### Weak assumption 3: current symmetric MLP pair features are probably sufficient

They are likely underpowered. Use low-rank biaffine interaction, explicit relative geometry, exact tactical pair features, and optionally a small candidate-pair refinement transformer over admitted pairs.

### Weak assumption 4: tactical priors might poison learning

Tactical priors poison learning only if used as a hidden value heuristic or if they replace self-play. Used as **candidate admission, auxiliary labels, and exact terminal solvers**, they are essential. Hexo is a sudden-death connection game; ignoring exact 4/5-window logic is wasteful.

### Weak assumption 5: uniform pair sampling is enough for early exploration

Pure uniform over pairs is too diffuse. You need a mixture: true uniform for support, tactical for recall, factored for coverage, and policy for exploitation.

### Weak assumption 6: small-board curriculum is harmless

Small boards introduce edges/corners and alter the infinite-board distribution. Prefer tactical microstates embedded in the real infinite/radius-limited rule system. Avoid edge-based curricula unless used only as a negative ablation.

### Weak assumption 7: 4-windows and 5-windows are “equally dangerous” in all modeling respects

Both are win-in-one-turn, but they create different target geometry. A 4-window requires one exact pair. A 5-window creates a winning single cell with many irrelevant second fillers. The training target should reflect that difference through terminal equivalence classes and marginal auxiliaries.

### Weak assumption 8: wall-clock can be handled after algorithm design

It must be measured from the start. Pair MCTS will look excellent at fixed simulations and can still lose at fixed time if pair scoring or candidate generation is too expensive.

---

## 9. First ablations to run

### Ablation 1: search representation

Compare:

1. sampled pair MCTS
2. sequential afterstate MCTS
3. autoregressive pair policy + shallow search
4. factored proposal + pair-edge MCTS

Run both **fixed simulations** and **fixed wall-clock**. The expected outcome: pair MCTS wins fixed simulations; sequential may initially win wall-clock; optimized sampled pair MCTS should eventually win both.

### Ablation 2: candidate admission

Compare:

1. policy top-k only
2. policy + uniform
3. policy + uniform + tactical
4. policy + uniform + tactical + factored
5. same as 4 with Gumbel
6. same as 5 with progressive widening

Failure criterion: tactical candidate recall below 95% on immediate-win/block benchmark.

### Ablation 3: pair scorer

Compare:

1. current symmetric MLP
2. low-rank biaffine only
3. biaffine + relative geometry
4. biaffine + relative geometry + tactical pair features
5. candidate-pair refinement transformer

Metrics: pair policy KL, tactical top-k recall, arena Elo, inference milliseconds per root.

### Ablation 4: sparse target treatment

Compare:

1. raw sparse pair CE over visited pairs only
2. candidate-set CE including zero-visit admitted pairs
3. proposal-aware search prior + candidate-set CE
4. plus marginal single auxiliary
5. plus autoregressive auxiliary
6. plus 5-window terminal equivalence collapse

Failure criterion: policy entropy collapses early or uniform/tactical candidates stop receiving visits despite high Q.

### Ablation 5: tactical module role

Compare:

1. no tactical module
2. terminal win/block admission only
3. terminal + cover-pair admission
4. full tactical proposal + auxiliary labels
5. tactical policy override beyond terminal states

Expected result: 4 should win; 5 may overfit or distort non-tactical play.

### Ablation 6: replay balancing

Compare uniform replay against stratified replay by:

* move number
* legal pair count
* tactical density
* root entropy
* search surprise
* value margin
* high regret
* proposal source

Success means better tactical benchmark performance without losing opening strength.

### Ablation 7: D6 correctness

Train with:

1. no symmetry
2. data augmentation only
3. augmentation + consistency loss
4. augmentation + test-time D6 ensemble

Failure criterion: transformed positions produce different best pairs or different tactical labels beyond a tiny tolerance.

---

## 10. Success metrics

Use four scorecards, not one.

### Playing strength

* Arena Elo versus previous checkpoints
* win rate versus sequential baseline
* win rate versus pure tactical engine
* win rate at fixed simulations
* win rate at fixed wall-clock

### Tactical competence

* immediate 4-window win recall
* immediate 5-window win recall
* forced block recall
* no-cover overload detection
* multi-axis fork detection
* pair-block hitting-set accuracy

### Search health

* candidate recall of high-Q pairs
* root policy entropy by phase
* proposal-source visit distribution
* KL between prior and search target
* Q gap between best admitted and best audit candidate
* percentage of roots where tactical module force-admitted the final move

### Systems efficiency

* nodes/sec
* neural inferences/move
* candidate scoring time
* root candidate count versus strength
* wall-clock Elo per millisecond
* memory per search tree

---

## Final concrete path

Build **Sampled-Gumbel Pair AlphaZero** with exact tactical admission and a graph/window encoder. The MCTS edge is the unordered pair. The pair policy is primary. Single-cell and autoregressive policies are auxiliary and proposal-only. Candidate admission is a logged mixture of model, tactical, factored, and nonzero uniform sources. Search uses proposal-aware priors, Gumbel admission at the root, and progressive widening at deeper nodes. Training uses sparse candidate-set pair targets, marginal/AR auxiliaries, child-Q/regret auxiliaries, D6 consistency, tactical labels, and replay balancing.

The one-sentence rejection: **do not make sequential afterstates the main algorithm; they are a useful wall-clock baseline and candidate generator, but they encode the wrong decision object for Hexo.**

[1]: https://openreview.net/forum?id=bERaNdoegnO "Policy improvement by planning with Gumbel | OpenReview"
[2]: https://link.springer.com/chapter/10.1007/978-3-642-25566-3_32 "Continuous Upper Confidence Trees | Springer Nature Link"
[3]: https://cdn.aaai.org/ojs/9661/9661-13-13189-1-2-20201228.pdf "Factored MCTS for Large Scale Stochastic Planning"
[4]: https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html "Graph Networks as Learnable Physics Engines for Inference and Control"
[5]: https://arxiv.org/abs/1810.00825 "[1810.00825] Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
[6]: https://arxiv.org/abs/1611.01734 "[1611.01734] Deep Biaffine Attention for Neural Dependency Parsing"
[7]: https://arxiv.org/abs/1506.03134 "[1506.03134] Pointer Networks"
[8]: https://gwern.net/doc/reinforcement-learning/model/alphago/2018-silver.pdf "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
[9]: https://www.sciencedirect.com/science/article/abs/pii/S0950705112001293 "Bitboard knowledge base system and elegant search architectures for Connect6 - ScienceDirect"
[10]: https://www.researchgate.net/publication/220437214_Two-Stage_Monte_Carlo_Tree_Search_for_Connect6 "Two-Stage Monte Carlo Tree Search for Connect6 | Request PDF"
