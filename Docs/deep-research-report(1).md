# Recommended Architecture for Hexo-RL

Hexo is not just “Gomoku on hexes.” The combination of an unbounded hex board, six-in-a-row win condition, radius-limited legality, and—most importantly—two placements on every post-opening turn means the strategically natural decision object is often a *pair* of cells, not a single cell. In your context document, that shows up directly in 4-window/5-window pressure, pair blocking, overload, shared pivots, and D6 symmetry handling. That framing is correct, but several parts of the current plan are still too optimistic about sparse joint-pair learning and too undecided about when to factor the action. fileciteturn0file0

## Bottom line

The best path is **not** “full exhaustive joint-pair MCTS” and it is **not** “fully sequential search over two single placements.” The best path is a **hybrid with pair actions as the search-semantic object, but sequential/autoregressive factorization inside candidate generation and policy learning**. In concrete terms:

Use an **AlphaZero-style search with the real simulator**, where the tree edge is an **unordered joint pair action** after the opening, but candidate pair admission is produced by a **single-cell policy + conditional second-cell policy + tactical proposal module + non-vanishing uniform floor**. Search those admitted pair actions with **Gumbel-style root admission / simple-regret search** and **progressive widening** deeper in the tree. Train the model with **joint-pair targets as the main policy target**, but **do not rely on them alone**: pair marginals, conditional second-choice targets, tactical auxiliary labels, and sampled negative/ranking losses should be first-class stabilizers, not minor extras. citeturn10view2turn16view0turn11view2turn11view0turn10view7turn0search5

If I had to reduce this to one sentence: **search should think in pairs, but the network should *propose* pairs through factorized structure and *score* them with a symmetric pair reranker.** That is the safest route to superhuman strength under both fixed-simulation and fixed-wall-clock evaluation. At fixed simulations, this preserves the real decision semantics; at fixed wall-clock, it avoids the quadratic disaster of naïve pair enumeration. citeturn16view0turn11view2turn17view1

I would **reject MuZero as the main research direction here**. Hexo has a fast exact simulator in Rust, so learned dynamics buy you little and introduce model error where you least want it. MuZero shines when the environment dynamics are unknown or expensive; in exact board games, AlphaZero-style planning with the true simulator is the cleaner baseline, and later analyses of MuZero indicate its learned models do not generalize equally well to unseen policies, with part of its strength coming from search being biased toward actions where the model is already accurate. That is the opposite of what you want in an enormous, tactically brittle pair-action space. citeturn10view2turn10view3turn0search5

## Search architecture

The central search decision for Hexo is whether the tree should branch on `(q1,r1,q2,r2)` or on a first placement followed by a second-placement afterstate. My recommendation is: **the tree policy should branch on the completed unordered pair**, with one special opening exception for Black’s initial single move. The reason is not aesthetic. It is strategic. In Hexo, a 4-window is already essentially a one-turn winning threat because both empties can be filled in the same move, and most decisive choices are synergistic pair choices: “create two windows that overload the opponent’s two blocks,” “block two distinct hot windows,” “convert one pivot plus one extender into multi-axis pressure,” and so on. A tree that treats the first stone as a first-class ply and the second stone as merely a follow-up distorts the real atomic action and encourages the search to overvalue single-stone saliency instead of pair synergy. fileciteturn0file0

That said, **naïve joint-pair MCTS is the wrong implementation**. Gumbel AlphaZero was motivated by the exact regime Hexo lives in: many actions, few simulations, and the need for policy improvement even when the search cannot visit everything. The paper shows that standard AlphaZero-style root action selection can fail to improve the policy when the number of actions exceeds the simulation budget, and it replaces that with Gumbel sampling without replacement plus simple-regret-oriented selection. Sampled MuZero then extends the same basic philosophy to complex action spaces by showing that sampled subsets can work *if* the sampling process is accounted for in planning and policy evaluation. Those are the two papers that matter most for Hexo’s branching problem. citeturn16view0turn13view1turn11view2turn11view0

So the right search stack is this:

At each node, maintain the game state as usual, but do **not** enumerate all legal pairs. Instead, construct a **candidate pair set** from a mixed proposal distribution:
- a model-driven single-cell prior,
- a conditional second-cell proposal given the first,
- a tactical generator that explicitly proposes immediate wins, urgent blocks, pair covers, overload-creating pairs, and shared-pivot forks,
- and a nonzero uniform legal-pair floor that never fully disappears. citeturn11view2turn11view0turn16view0turn6view0turn8search5

At the **root**, use **Gumbel-Top-k admission without replacement** over either pair candidates directly or over first-cell candidates followed by conditional second-cell completion. Gumbel matters here because the root is where the training target is produced and where catastrophic support collapse begins if you let a weak policy simply feed top-k candidates back into itself. Gumbel’s policy-improvement argument is precisely about this risk. citeturn16view0turn13view2

Below the root, I would use **progressive widening over pair children** rather than full Gumbel at every node. Progressive widening is the correct control knob when the branching factor is effectively combinatorial. The older progressive-widening literature is about continuous and stochastic spaces, but the core lesson applies here: if you widen too fast, you get a wide, shallow tree with too few visits per child; if you widen too slowly, you miss good actions. Hexo’s pair space is discrete, but it behaves like a structured large-action domain, so this tradeoff is real. The recent state-conditioned action-abstraction paper makes the same point in a MuZero-like large combinatorial action setting: MCTS degrades sharply when an action is composed of sub-actions and the search fails to exploit compositional structure. citeturn17view1turn10view7

The subtle but important point is this: **the pair is the backed-up action, but internal statistics may still be factored.** You can cache first-cell statistics, second-cell conditionals, and tactical cover-set summaries to make child admission cheaper. That is useful engineering and useful inference structure. But the *backup edge* should still be the full pair child. Do not let factored statistics replace pair-level value estimates. They are a proposal accelerator, not the semantic action of the game. citeturn10view7turn11view2

For tactically sharp states, I would also add a **specialized tactical subsearch trigger** rather than relying on general MCTS alone. Classical Connect6 work is directly relevant here because Connect6 has the same post-opening two-stone turn structure and abrupt “sudden death” tactical landscape. Relevance-Zone-Oriented Proof Search solved many Connect6 tactical positions and openings by focusing on threat-based regions, and Two-Stage MCTS for Connect6 reported that tactical subsearch/threat-space components substantially improved efficiency in positions with tactical solutions. In Hexo, I would not make that the general planner, but I *would* use a small threat-space or proof-style module whenever the state contains immediate 4-window/5-window races or multi-threat overload clusters. citeturn6view0turn8search5

The blunt comparison is:

- **Full pair MCTS, exhaustive children:** strongest semantics, worst wall-clock, reject.
- **Sequential afterstate MCTS:** best cheap baseline, but strategically misaligned if used as the main search tree.
- **Pair children with autoregressive/factored proposal and widening:** best overall research path.
- **MuZero with pair macro-actions:** unnecessary complexity unless your simulator becomes the bottleneck, which it should not in Hexo. citeturn10view2turn10view3turn16view0turn11view2

## Model architecture

The main encoder should be a **global graph/set model**, not a fixed-crop CNN. Your context document is correct on this point. Hexo’s board is unbounded, legal actions are variable-length sets of global coordinates, and pair rows are references to legal action rows rather than natural pixels. A fixed crop can still be a useful ablation, but it should not be the flagship architecture. Deep Sets gives the basic argument for permutation-invariant/equivariant set processing, and Set Transformer gives the more relevant one for your case: set elements can interact through attention while remaining permutation-invariant, with scalable attention variants for larger sets. That maps much better onto “stones + legal cells + tactical windows” than a crop-based policy head does. citeturn12view0turn10view4

The encoder I would build is a **sparse relational transformer over variable tokens**:
- one or a few global state tokens,
- stone tokens with color, age/ply, and relative axial coordinates,
- legal single-cell tokens for all legal placements,
- and **window tokens** for tactically active six-cell windows only, not every possible geometric window. fileciteturn0file0

That last choice is crucial. If you want the model to learn Hexo-specific tactics—4/5-window threats, pair blocking, overload, shared pivots, multi-axis pressure—you should not hope that it rediscovers all of that purely from raw spatial adjacency. The model should see explicit *objects* for those windows. Window tokens should carry axis identity, occupancy counts, whether they are “hot,” which empties matter, and which legal cells participate in them. Then legal-cell embeddings can aggregate over the windows they touch, and global state embeddings can aggregate over the current tactical landscape. That is the cleanest way to make “threat overload” a learnable relational object instead of a vague emergent pattern. fileciteturn0file0

For D6 symmetry, I would **prefer aggressive exact augmentation + symmetry consistency tests/losses over trying to force perfect D6 equivariance into every layer from day one**. Hexagonal lattices do have natural 6-fold rotational structure, and hexagonal/group-convolution work shows that this symmetry can be used effectively, but your action space is not just a dense lattice output—it is a variable legal-row set plus unordered pair rows. That makes exact end-to-end equivariant implementation possible but easy to get wrong. The safer path is: random D6 augmentation every sample, exact transformed row-table validation, symmetry consistency penalties on policy and value, and only then explore stricter equivariant layers if ablations justify the complexity. citeturn4search5

For the **policy heads**, I would use a three-part design:

First, a **single-cell unary head** over legal cells. This is not the final action policy, but it is essential. It supplies `p(first | s)`-like structure, dense supervision through pair marginals, tactical saliency, and a reusable proposal prior. Pair-only learning is too sparse to give this up. citeturn11view2turn16view0

Second, a **conditional second-cell pointer head** over `p(second | s, first)`. Pointer Networks are relevant here because they were designed for variable-sized output dictionaries over input positions. Hexo’s legal-cell set is variable, and the “second stone conditioned on a chosen first stone” is exactly the kind of conditional selection problem a pointer-style head naturally models. This head is the right workhorse for candidate generation and for densifying learning targets. citeturn10view5

Third, a **symmetric pair reranker** over candidate pairs. This should *not* be only an MLP on `[h_i + h_j, |h_i-h_j|, h_i ⊙ h_j]`, though those features are useful. The strongest version is a **biaffine or low-rank bilinear interaction term plus symmetric MLP features and explicit pair-tactical features**:
\[
E(i,j) = b_i + b_j + h_i^\top U h_j + \mathrm{MLP}(s, h_i+h_j, |h_i-h_j|, h_i\odot h_j, g_{ij}, t_i, t_j, t_{ij})
\]
where `g_ij` encodes geometry and `t_ij` encodes pair-specific tactical relations such as joint cover count, immediate-completion count, overload score, shared-pivot participation, and axis diversity. Biaffine scoring is relevant because it is a proven way to model asymmetric or pairwise interactions efficiently without turning every pair into a full token. In Hexo I would make it symmetric at the final unordered-pair level, but keep the low-rank bilinear interaction because pair synergy is the entire point. citeturn3search3turn12view0turn10view4

What I would **not** do is materialize all legal pairs as graph tokens inside the main encoder. Your own document is right to worry about `O(A^2)` blowup. Pair scoring belongs in the head over candidates, not in the attention sequence. fileciteturn0file0

The value side should include the main state value plus at least two tactical auxiliaries:
- an **immediate tactical resolution head** for “forced win / forced loss / stable / unclear within a short tactical horizon,”
- and a **pair-regret or completed-Q head** over searched candidates.  
This is directly inspired by Gumbel’s “completed Q” idea: if large portions of the action space are unvisited, you need a principled baseline for them rather than implied zeros. citeturn16view0

## Training and exploration

The biggest training mistake you can make is to say: “the main policy is over joint pairs, so let’s only train on sparse joint-pair MCTS visit targets.” That is *too sparse* for the early and middle phases of learning. Sampled MuZero shows that sampled action subsets can still train meaningful global policies, but only with careful bookkeeping about how the subset was sampled and how the search prior is corrected. Gumbel also shows that improved policy targets should be constructed from more than raw visits alone when actions are only partially explored. In Hexo, this means sparse joint-pair targets can work, **but only inside a thicker training scaffold**. citeturn11view0turn11view4turn16view0

The training policy loss should therefore have four layers:

The **main loss** is on the **candidate-restricted improved joint-pair policy** coming out of search, over the admitted candidate set. This is still the main action target because the true decision is the pair. citeturn16view0turn11view2

The first stabilizer is a **pair-marginal single-cell loss**:
\[
\pi_{\text{cell}}(i) = \sum_j \pi_{\text{pair}}(i,j)
\]
This should not be a cosmetic auxiliary. It should be a major loss component throughout training, because it gives dense coverage over legal cells even when the pair target is sparse. fileciteturn0file0

The second stabilizer is a **conditional second-cell loss** from the joint target:
\[
\pi(j \mid i) \propto \pi_{\text{pair}}(i,j)
\]
for first cells that received meaningful target mass. This is the loss that makes the autoregressive proposal head actually useful instead of merely convenient. citeturn10view5turn11view2

The third stabilizer is a **sampled negative/ranking loss** on unvisited or low-value candidate pairs. Do not treat all unadmitted pairs as negative—those are unknown. But within the admitted set, and among separately sampled negatives, a ranking loss is important to prevent the pair scorer from only learning from positives. This is one of the weakest points in your current plan. Sparse-positive-only pair training will be too forgiving and will overfit the proposal support. citeturn11view0turn16view0

For search bias, I would explicitly log:
- the candidate set,
- the proposal source of each candidate,
- and the proposal probability or mixture weight.  

Then use a **Sampled-MuZero-style corrected prior inside search** rather than naïvely plugging the raw model pair prior into PUCT over sampled candidates. The key Sampled MuZero result is exactly that “planning over a sampled subset” is not automatically principled; the sampling must be reflected in the search prior, and the paper reports that using the corrected prior is materially more robust than using the raw prior. That idea transfers directly to Hexo’s pair-candidate sampling problem. citeturn11view0turn11view4

Exploration should be phased.

In the **bootstrap phase**, do *not* rely on pure random self-play with pair-only targets. Instead, use a **tacticalized high-entropy proposal mixture** with a large uniform floor and a generous tactical sampler. The goal here is not to inject brittle human heuristics as the policy; it is to guarantee that the model actually sees winning pairs, urgent block pairs, and overload motifs at all. Gumbel-style sampling without replacement helps at the root, but it is not magic if the proposal support is terrible. citeturn16view0turn13view1turn6view0turn8search5

In the **growth phase**, anneal toward model-led candidate admission, but never eliminate the uniform floor. Your document is correct here. Deterministic top-k from an immature policy is how support collapse becomes permanent. I would keep a small but real uniform pair sampler throughout training, and I would keep tactical proposals alive much longer than most AlphaZero systems keep hand-designed helpers, because Hexo’s tactical cliff edges are so sharp. fileciteturn0file0turn16view0turn11view2

In the **mature phase**, lower action temperature, increase search budget gradually, and use progressive simulation-budget growth. The MiniZero board-game study is relevant here because it found that progressively increasing the simulation budget during training improved performance in board-game settings. That is a better curriculum than changing the board size or legality radius, because it preserves the action semantics of the real game. citeturn0search2

Replay should *not* be uniform over positions. I would stratify or prioritize by:
- game phase,
- tactical density,
- search surprise `KL(search || policy)`,
- pair-space size,
- and short-horizon regret / value uncertainty.  

This is especially important because the most strategically important Hexo positions are not evenly distributed. Uniform replay will overrepresent boring early states and underrepresent rare, overloaded tactical states where pair reasoning matters most. That is an inference from your game structure and from the sampled-action bias problem, and it should be treated as a core design decision, not a tuning footnote. fileciteturn0file0turn11view0

## Hexo-specific learning and the assumptions to reject

Hexo-specific strategy should be learned through **explicit tactical object representations**, not only through generic value/policy supervision. The model needs to reason about six-cell windows, hot 4-windows and 5-windows, block-cell sets, cells shared across multiple windows, and axis interactions. The clean way to do that is:
- encode hot window objects,
- provide dense tactical labels derivable from exact rules,
- use those labels as auxiliary supervision,
- and use the same tactical objects for candidate proposal and tactical benchmark construction. fileciteturn0file0turn6view0turn8search5

The tactical proposal module should propose, at minimum:
- immediate winning pairs,
- immediate forced blocks,
- minimal covers of multiple opponent hot windows,
- overload-creating own pairs,
- and pivot/extender pairs that increase independent hot-window count across distinct axes.  

These should be **proposal sources and auxiliary targets**, not hard action filters. Hard tactical filters are dangerous because they can poison long-horizon learning and make the policy depend on brittle handcrafted logic, but *proposal modules* are exactly where classical Connect6 knowledge belongs. Classical Connect6 search succeeded not because it hand-coded the whole game, but because it injected threat-space structure where the combinatorics became tactically sharp. That is the right lesson to port. citeturn6view0turn8search5

The assumptions I would explicitly reject or downgrade are these:

The assumption that **joint-pair targets alone should define the main policy and singles should stay auxiliary-only** is too extreme. Joint pairs should remain the main action target, but pair marginals and conditional-second targets should be treated as *major training channels*, not ornamental helpers. fileciteturn0file0turn11view2turn16view0

The assumption that **sequential search is a serious long-term alternative to pair-level search** should be downgraded. It is a necessary baseline and a useful internal factorization, but it should not be your flagship planner unless equal-wall-clock experiments decisively show otherwise after the pair system is mature. Early wins by sequential search will often be an artifact of lower expansion cost, not better strategic semantics. citeturn16view0turn10view7

The assumption that **current symmetric MLP pair features are likely enough** is weak. They are a valid baseline, but a pair game with strong cross-cell synergy is exactly where a biaffine or low-rank bilinear interaction should pay for itself. citeturn3search3

The assumption that **small-board or restricted-action curricula are likely helpful** should be treated skeptically. They change the action distribution and the geometry of overload in exactly the way you do *not* want. The safer curriculum is over search budget, candidate budget, replay mix, and tactical supervision—*not* over a different game. fileciteturn0file0turn0search2

The assumption that **MuZero deserves equal billing as a top candidate** is wrong for this project. With a perfect simulator, it should be a side curiosity, not the main plan. citeturn10view2turn10view3turn0search5

## First experiments and success criteria

Your first experiments should not be broad. They should answer the pair-action question as quickly and cleanly as possible.

The highest-priority ablation is **search semantics under equal simulation and equal wall-clock budgets**:
- pair-macro search with sampled candidate pairs,
- sequential afterstate search,
- and pair-macro search with autoregressive candidate generation.  
Use the same backbone and value head for all of them. Measure Elo, tactical benchmark accuracy, root-policy improvement, and candidate recall at both fixed simulations and fixed milliseconds per move. This is the experiment that decides whether pair-as-action is practically right, not just philosophically right. Use a mature enough training run that proposal quality is not the bottleneck. citeturn16view0turn0search2turn16view0

The next ablation is **pair head design**:
- symmetric MLP pair features only,
- autoregressive only,
- biaffine/symmetric reranker only on uniformly sampled candidate pairs,
- and the recommended **autoregressive proposal + symmetric biaffine reranker**.  
The winning metric here is not just policy loss. It is top-k pair recall against deep search or tactical oracle labels, plus equal-time arena strength. If the biaffine reranker does not raise pair recall on overloaded tactical states, abandon it quickly. citeturn3search3turn10view5

The third ablation is **candidate-set construction**:
- policy-only,
- policy + uniform,
- policy + tactical,
- and policy + tactical + uniform + factored composition.  
This experiment tells you whether your search is collapsing due to proposal support. Track how often the best searched pair came from each proposal source and how often the final chosen action would have been unavailable without each source. If the uniform floor still contributes meaningful winning candidates late in training, keep it permanently. citeturn11view0turn16view0turn6view0turn8search5

The fourth ablation is **target structure**:
- joint-pair loss only,
- joint + cell marginal,
- joint + cell marginal + conditional second,
- and the full package plus sampled ranking negatives.  
The main readouts are learning stability, policy entropy, candidate recall, tactical benchmark improvement, and whether replay collapses onto a narrow action-support mode. I expect joint-only to lose badly in early and mid training. citeturn11view0turn16view0

The fifth ablation is **tactical representation**:
- no window tokens,
- hot-window tokens only,
- hot-window tokens + dense tactical auxiliaries,
- and hot-window tokens + tactical proposal module.  
If Hexo-specific strategy really is as threat-object-driven as your document claims, this ablation should show a large improvement in immediate-win/block accuracy and in equal-time arena strength in tactical midgames. If it does not, your tactical representation is probably too shallow or incorrectly wired. fileciteturn0file0turn6view0turn8search5

The minimal success criteria should be:
- pair-macro model beats sequential baseline at equal wall-clock on mature checkpoints,
- candidate recall of strong pairs stays high under widening,
- no severe support collapse in pair-policy entropy,
- tactical test-suite accuracy on immediate win/block and overload states is high,
- and D6 transforms preserve row identities and policy/value consistency exactly up to numerical tolerance. fileciteturn0file0turn16view0turn11view0

The failure criteria should be equally clear:
- pair-macro search only wins at equal simulations but loses at equal time,
- the best pair frequently lies outside the admitted candidate set,
- joint-only training becomes overconfident with poor tactical recall,
- or tactics improve only when the tactical proposer is present, indicating the network itself is not internalizing the game. citeturn17view1turn11view0

## Why these papers actually apply

**AlphaZero** applies because Hexo has a perfect simulator and the core problem is approximate policy iteration with search, exactly the setting AlphaZero was built for. citeturn10view2

**Gumbel AlphaZero / Gumbel MuZero** applies because Hexo lives in the regime where actions greatly outnumber simulations, and the paper’s main contribution is a policy-improving way to search when you cannot visit everything. That is the exact failure mode of naïve pair candidate admission. citeturn16view0turn13view1

**Sampled MuZero** applies because your pair space is a sampled structured action space. Its most useful contribution here is *not* learned dynamics; it is the theory and practice of planning over a sampled action subset with corrected priors and careful bookkeeping. That is directly portable to sampled pair candidates in Hexo. citeturn11view2turn11view0turn11view4

The recent **state-conditioned action-abstraction** work applies because Hexo actions are explicitly compositional—two coordinated sub-actions per turn. Its message is that large-action MCTS should exploit sub-action structure on the fly, which is exactly why I recommend factorization for proposal and caching, but not for the backed-up semantics of the edge. citeturn10view7

**Relevance-Zone-Oriented Proof Search** and **Two-Stage MCTS for Connect6** apply because Connect6 is the closest classical analogue: same “single opening stone, then two placements per turn” structure, same sudden-death threat geometry, same importance of sharp tactical subsearch in a gigantic branching factor. They support adding a tactical threat-space module and relevance-zone logic for Hexo’s hot-window races. citeturn6view0turn8search5

**Deep Sets** and **Set Transformer** apply because your core objects are variable sets—stones, legal cells, windows, candidate actions. They justify a permutation-aware variable-length encoder and specifically support using attention to model interactions among action-relevant objects without committing to a fixed board crop. citeturn12view0turn10view4

**Pointer Networks** apply because “pick a second legal cell conditioned on the first” is naturally a variable-dictionary selection problem over a legal-action set. That is exactly the role of the conditional second-cell head in the recommended architecture. citeturn10view5

**Biaffine scoring** applies because Hexo’s policy quality depends on modeling pair interactions, not just unary cell quality. The point is not to import NLP methods wholesale; it is to use a proven efficient form for pairwise interactions without blowing up the whole model into pair tokens. citeturn3search3

**MiniZero’s progressive simulation** result applies because your curriculum should vary search compute more than game structure. It is evidence that progressively increasing simulation budget can improve board-game training efficiency, which is much safer than changing Hexo’s actual action structure. citeturn0search2

The papers that matter least here are the pure **MuZero learned-dynamics** extensions. Their relevance is much weaker because Hexo already has exact rules and a fast simulator, and later analysis has shown MuZero’s learned models are not uniformly robust outside the policies they are trained around. citeturn10view3turn0search5

## Open questions and failure conditions

The most important unresolved question is not *whether* the search edge should be a pair. I think it should. The unresolved question is **how much of the pair machinery should happen before search versus inside search**—that is, the optimal split among proposal quality, reranker quality, and widening schedule. citeturn11view2turn17view1

The second open question is whether **interior-node Gumbel** is worth its wall-clock cost or whether **root Gumbel + interior widening/PUCT** is the better trade. The literature strongly supports Gumbel at low-simulation roots, but Hexo’s branch explosion at all nodes means this needs a direct equal-time ablation rather than faith. citeturn16view0turn17view1

The third is whether **strict D6-equivariant encoders** beat **aggressive symmetry augmentation plus consistency loss** once the full pair-row machinery is included. I would start with the latter because it is safer and easier to validate. citeturn4search5turn12view0

If those open questions are handled well, the decisive recommendation still stands: **build Hexo-RL around sampled joint-pair MCTS with pair-level backups, autoregressive candidate generation, a symmetric biaffine pair reranker, explicit hot-window tactical objects, and a training stack that treats sparse pair targets as primary but not sufficient.** That is the strongest and safest architecture I would bet on for this game. citeturn16view0turn11view2turn6view0turn8search5turn12view0turn10view5