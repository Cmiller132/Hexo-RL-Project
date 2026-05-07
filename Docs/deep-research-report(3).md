# Pair-Action Policy Priors for Hexo-RL MCTS

## Executive judgment

Hexo-RL is an entity["academic_field","Reinforcement Learning","machine learning field"] problem with a combinatorial action space inside entity["scientific_concept","Monte Carlo Tree Search","search algorithm"]. After the opening, a move is an unordered pair of empty cells, and because each turn places two stones, both 5-windows and 4-windows are win-in-one-turn motifs. That combination makes pair recall and tactical safety stricter than in ordinary single-move entity["software","AlphaZero","self-play game-playing algorithm"]-style policies. fileciteturn0file0 citeturn27view0turn17view1turn13search0

The most promising architecture is **a factorized pair policy with dense cell marginals plus a sparse symmetric pair-correction scorer over retrieved candidate edges**, fed to search as a **candidate-set conditional prior** rather than as a globally normalized policy over all \(\binom{A}{2}\) pairs. Concretely: keep a shared legal-cell encoder; predict a dense marginal score for every legal cell; retrieve a modest number of promising pair edges using low-rank symmetric retrieval; rerank those edges with a symmetric correction head; and normalize exactly over the admitted candidate set. This design gives you dense supervision through cell marginals, subquadratic inference, legality and unordered symmetry by construction, and much better debuggability than your current Plan A. The next two architectures I would seriously build are **a latent mixture-of-soft-pointers policy** and **a symmetrized autoregressive unordered-pair policy with ANN completion**. I would treat DETR-like set decoders and learned generative widening samplers as second-wave experiments, not as the first production candidate. citeturn19view3turn31view0turn20view0turn18view3turn20view1turn6search3

The key strategic conclusion from the literature is that you do **not** need a faithful full-action softmax over all unordered pairs. Sampled-action planning already has a principled foundation. Sampled MuZero explicitly plans over sampled action subsets, Gumbel planning fixes the root-policy-improvement pathology that appears when not all actions are visited, and the regularized-policy view of MCTS makes it natural to think in terms of relative priors on the actions that search actually admits. In parallel, entity["software","KataGo","Go engine and training system"] shows that search exploration and training targets should be decoupled, and that dense auxiliary targets often matter more than squeezing one more clever heuristic into the proposer. citeturn13search0turn17view1turn23view0turn28view0

## Domain constraints that should drive the architecture

Hexo’s board is an infinite axial hex grid; after the opening, a legal move is an unordered pair of empty cells within placement radius 8 of any existing stone; wins are lines of 6 on the three hex axes; and the natural board symmetry group is **D6**, with 12 transforms and axis permutation under rotation. Those facts mean the action head must be variable-cardinality, legality-masked, canonicalized by board coordinates rather than by transient legal-list order, and tested for D6 consistency at the **pair** level, not only at the cell level. fileciteturn0file0

The two-stone turn rule changes the search problem more than it changes the network problem. A missed immediate pair is catastrophic, so a tiny deterministic tactical override for exact win-in-one-turn and must-block pairs is justified regardless of the learned architecture. In Hexo’s own terminology, 5-windows and 4-windows are equally urgent because both can be completed in one turn, and unblockable positions arise from stacking more independent hot windows than two placements can cover. That is a strong argument for separating “don’t miss the tactic” from “rank the large residual action space well.” fileciteturn0file0

This also changes what the policy target should be. Using raw search visits as “the” target is the wrong abstraction when search intentionally injects noise or forced exploration: KataGo’s policy-target pruning exists precisely because exploratory visits should help discovery without being copied wholesale by the apprentice, and its auxiliary policy, ownership, and score targets materially improved training efficiency. For Hexo, the dense analogue is immediate: if search assigns mass \(q(\{i,j\}\mid s)\) to pair actions, then the cell marginal  
\[
q_{\text{cell}}(i\mid s)=\sum_{j\neq i} q(\{i,j\}\mid s)
\]
is a dense, informative target available at every node, even when pair supervision is sparse. That target is closer to “KataGo-style dense supervision” than any pair-only objective. citeturn28view0

Backbone choice is mostly orthogonal to the pair head, but it still matters. A legal-cell encoder should either be a board network with global context injection, in the spirit of KataGo’s global pooling, or an explicitly variable-set encoder such as a Set Transformer. The point is the same in both cases: local geometric motifs matter, but whole-position information such as threat multiplicity, remaining block coverage, and board-scale asymmetry also matters, and variable legal sets should not force you into brittle fixed-output tensors. citeturn28view0turn18view0

## Factorized marginal with sparse symmetric correction

The practical architecture I would start with is a **three-part pair policy**:

\[
\ell_{ij}=m_i+m_j+\delta_{ij}, \qquad i<j,
\]

where \(m_i\) is a dense marginal logit for legal cell \(i\), and \(\delta_{ij}=\delta_{ji}\) is a sparse symmetric correction evaluated only on a candidate edge set \(E(s)\), not on all legal pairs. This is the cleanest realization of the “low-rank energy model with sampled normalization” family: factorized models are attractive under sparsity because they share statistical strength across pair interactions, and large-action RL systems already use retrieval-then-rerank pipelines to avoid exhaustive action scoring. citeturn19view3turn17view9turn31view0

A concrete instantiation is to let the encoder produce legal-cell embeddings \(h_i\in\mathbb{R}^d\). The marginal head gives \(m_i=w^\top h_i\). The correction head uses a state-conditioned low-rank symmetric score such as
\[
\delta_{ij}=\langle R_s h_i, R_s h_j\rangle + \psi(h_i\odot h_j, g_{ij}, s),
\]
where \(g_{ij}\) are cheap geometric pair features such as same-axis, distance, shared-hot-window count, or block-set overlap, and \(\psi\) is a small MLP. The retrieval stage uses the transformed embedding \(R_s h_i\) to fetch promising partners \(j\) by approximate nearest-neighbor or MIPS-style lookup, exactly the kind of sublinear candidate generation used in large discrete action spaces and large-corpus retrieval. The network output is therefore \(A\) dense marginals plus \(C\) sparse pair scores, not \(A^2\) pair logits. citeturn31view0turn24view0turn20view0

This head gives a simple candidate generator. Take the top \(K\) cells by \(m_i\). Form a **small** cross-product subset from them, because some good pairs really are “two individually strong cells.” Then, for each of those \(K\) anchors, retrieve top partners using the symmetric edge key. Add exact tactical pairs from the deterministic oracle. Deduplicate by canonical pair identity \((i,j)\) with \(i<j\) under a state-independent board-coordinate ordering, mask illegals, and rerank. Search then receives
\[
P_{\text{MCTS}}(\{i,j\}\mid s,E)=\frac{e^{\ell_{ij}/\tau}}{\sum_{(u,v)\in E} e^{\ell_{uv}/\tau}},
\]
which is **exactly normalized over admitted candidates**. That is the right object for MCTS. You should not expose sampled-softmax or NCE scores directly to search once the candidate set is already small enough to normalize exactly. citeturn13search0turn17view1turn23view0turn18view3turn20view1turn6search3

Training is the main advantage of this design. Use the dense policy target \(q_{\text{cell}}\) for \(m_i\), and train the pair correction only on a candidate-restricted set:
\[
\tilde q_E(p\mid s)=\frac{q(p\mid s)}{\sum_{u\in E(s)} q(u\mid s)}, \qquad p\in E(s).
\]
Then use candidate-restricted cross-entropy on \(E(s)\). Pairs outside \(E(s)\) are **unknown**, not negatives. For the retrieval stage itself, use sampled-softmax or entity["scientific_concept","Noise-Contrastive Estimation","estimation method for unnormalized models"] against proposal-distributed negatives; for the final reranker, use exact candidate-softmax. If you want a ranking-style auxiliary, BPR is the right mental model: missing or unproposed pairs should not be globally labeled 0, and sampled pairwise comparisons better match the actual ranking objective than dense one-vs-all classification. citeturn18view3turn20view0turn20view1turn6search3turn26view0

The compute profile is favorable. Ignoring the shared backbone, the marginal head is \(O(A d)\). Top-\(K\) anchor selection is \(O(A\log K)\) or faster with selection primitives. Retrieval is roughly \(O(K\log A)\) for ANN lookup, plus \(O(C d)\) for reranking the final candidate budget \(C\). If you also include a capped cross-product among top marginals, that adds at most \(O(K^2)\), but in practice you cap it aggressively and let \(C\) dominate. The crucial point is that the architecture never requires computing all \(A^2\) pair scores. citeturn31view0turn24view0

Unordered symmetry is exact because \(\ell_{ij}=\ell_{ji}\). Legality is exact because you only encode legal cells and only create pairs with two distinct legal coordinates. D6 consistency can be added as
\[
L_{D6}=\mathbb{E}_{g\in D6} \,\mathrm{JS}\!\left(P_\theta(\cdot\mid g s),\, g\!\cdot\! P_\theta(\cdot\mid s)\right),
\]
with the action transformed under the same precomputed lookup tables you already use for augmentation. I would start with augmentation plus consistency testing before attempting a hard D6-equivariant architecture. fileciteturn0file0

The main failure mode is **hidden-synergy miss**: the best pair may contain two individually modest cells whose value lives mostly in \(\delta_{ij}\), so a generator dominated by top marginals can miss it. That is exactly the test that should decide whether you stay with this architecture or move to a latent-intent mixture. Relative to your Plan A, the architecture is much simpler and easier to audit because there are only two real proposal channels—dense marginals and low-rank pair retrieval—plus the tactical safety path. Relative to Plan B, it avoids needing partner logits for every anchor and replaces sparse pair-only supervision with dense cell marginals. citeturn28view0turn31view0turn19view3

The falsification test is straightforward: if, at matched latency and candidate budget, this architecture underperforms Plan A on best-pair recall or posterior-mass coverage specifically on states where the audit-best pair has **no endpoint inside the top-\(K\) cell marginals**, then the factorized decomposition is too strong and you should escalate to a latent-intent model.

## Latent mixture of soft pointers

The best alternative when you suspect many good pairs are **not** “strong cell + strong partner” is a **latent mixture-of-soft-pointers** model. Let the head output \(M\) latent intents with weights \(\pi_m(s)\), plus two pointer distributions over legal cells for each intent. The unordered pair distribution is
\[
p(\{i,j\}\mid s)=\sum_{m=1}^{M}\pi_m(s)\Big[\alpha_m(i\mid s)\beta_m(j\mid s)+\alpha_m(j\mid s)\beta_m(i\mid s)\Big],
\]
with same-cell events masked out and renormalized. The key idea is that each component can specialize to a different pair motif—extension pair, bridge pair, split-threat pair, block pair—without forcing those motifs to appear as globally high cell marginals. This borrows the expressivity argument from Mixture of Softmaxes and the variable-dictionary action selection mechanism from Pointer Networks. citeturn30view0turn18view1

Candidate generation is cheap if \(M\) is small. For each mixture component, keep only its top \(K\) support cells under \(\alpha_m\) and \(\beta_m\), form \(O(K^2)\) within-component candidates, and then deduplicate across components. With fixed \(M\), the head cost is \(O(M A d)\) for cell logits and \(O(M K^2)\) for proposal formation, which is still far below \(A^2\) when \(K\) is small. This is the main reason the mixture model is more interesting than Plan B’s all-anchor completion: it lets you spend proposal budget on **semantic modes**, not on every anchor cell. If you want a differentiable recall regularizer rather than only likelihood, the sparse top-\(k\) literature gives workable surrogates for training a model to care about which support makes it into the candidate set. citeturn30view0turn18view1turn21view1

Training can still avoid false negatives cleanly. Use pair likelihood on the search distribution and auxiliary dense cell targets from the induced marginal
\[
q_{\text{cell}}(i)=\sum_j q(\{i,j\}),
\]
just as in the recommended architecture. The pair likelihood marginalizes over latent components, so you are not forced to choose a single “correct component” during training. To prevent support collapse, add a **component load-balancing** term and penalize low entropy over mixture usage early in training. That recommendation is not arbitrary: the reason Mixture of Softmaxes works is precisely that different latent components can carry different high-rank structure, and the reason generative slate models help is that they model whole slates rather than independent items. citeturn30view0turn25view0turn25view2

MCTS integration is easy. Normalize the deduplicated candidate scores over the final candidate set and feed them into root Gumbel admission or standard PUCT. Unlike a set decoder, there is no fixed slot capacity, and unlike a plain autoregressive model, there is no single-anchor bottleneck. In my view, this is the best fallback if the factorized marginal+corrrection architecture repeatedly misses “two individually unremarkable but jointly decisive” pairs. citeturn17view1turn22view0

The risk is training stability. Mixture components can collapse into near-duplicates, calibration can drift if a few components dominate, and debugging gets harder because failure can arise from either poor cell distributions or poor mixture routing. Compared with Plan A, it is simpler at inference but less transparent. Compared with Plan B, it is more expressive and less tied to “every good pair has an obvious anchor,” but the calibration story is weaker unless you put serious effort into load balancing and candidate-set recalibration.

The decisive experiment is targeted: build a “low-endpoint-salience” benchmark consisting of states where the audit-best pair contains no cell in the top-10 audit cell marginals. If this architecture materially lifts best-pair recall and covered posterior mass on that slice without blowing up latency or calibration, it has earned its extra complexity. If not, the mixture is mostly expressive overhead.

## Symmetric autoregressive unordered pair policy

A symmetrized autoregressive policy is the cleanest probabilistic model of an unordered pair:
\[
p(\{i,j\}\mid s)=p_1(i\mid s)\,p_2(j\mid s,i)+p_1(j\mid s)\,p_2(i\mid s,j),
\]
where \(p_2(\cdot\mid s,i)\) is masked to legal cells other than \(i\). This identity is exact, naturally normalized, and gives MCTS a very interpretable pair prior. It is also a natural extension of Pointer Networks: the first pointer chooses one cell from a variable legal set, the second pointer chooses another, and unordered symmetry is restored analytically by summing both orientations. citeturn18view1

This architecture only becomes interesting if the conditional completion step is genuinely subquadratic. If \(p_2(j\mid s,i)\) requires scoring all \(A\) partners for every anchor \(i\), you are back to \(O(A^2)\) work. The practical version therefore uses ANN/MIPS-style completion: compute \(p_1\) over all legal cells, keep top \(K\) anchors, and let each anchor query a partner index over cell embeddings. You can then use Gumbel-top-\(k\) or stochastic beam search to sample or admit candidate ordered pairs without replacement, and collapse them to unordered pairs before reranking. The resulting cost is roughly \(O(A d + K\log A + C d)\), not \(O(A^2)\), provided the partner lookup is retrieval-based. citeturn31view0turn22view0

The best training recipe again uses dense targets. The first-stage head gets the dense search-derived cell marginal \(q_{\text{cell}}\). The second-stage head gets a conditional target
\[
q(j\mid i,s)\propto q(\{i,j\}\mid s),
\]
for anchors \(i\) with nontrivial search marginal mass. That is much better than supervising only the final pair identity. A further practical refinement is to follow KataGo’s lesson and prune exploration-only search visits before turning them into targets, so search noise helps discover partner completions without teaching the policy to imitate noise. citeturn28view0

This is still weaker than the recommended architecture for one reason: **anchor bias**. If neither endpoint is salient enough to survive the first-stage truncation, the pair is dead on arrival. Your Plan B already intuits this failure mode, and the literature does not remove it. What the literature does offer is a cleaner mathematical way to write the policy, better tools for sampling without replacement, and a principled sampled-action view of planning once the pair candidates have been generated. citeturn17view1turn13search0

I would only choose this model over the factorized marginal+corrrection architecture if the empirical completion distributions turn out to be extremely sharp. In that regime, the conditional model buys you crisp probabilities and elegant training. If completion is broad or multimodal, the sequential factorization just hides recall problems in the first stage.

The falsification test is to compare it directly against the factorized architecture on two slices: states where the best pair has one clearly dominant endpoint, and states where it does not. If it wins only on the first slice and loses overall on equal-wall-clock Elo, then it is the wrong default despite its clean probability model.

## Set decoders and learned widening samplers

A DETR-like set decoder is the main non-autoregressive alternative. It emits \(K\) query slots, each slot predicts two cell pointers and an objectness logit, unmatched slots predict “no pair,” and bipartite matching enforces one-to-one assignments during training. That is attractive because you directly optimize the object you care about—a bounded set of pair proposals—and DETR’s set loss is explicitly designed to avoid duplicates. For Hexo, each slot would output an unordered pair candidate and its prior, which search could consume after canonicalization and optional reranking. citeturn16search1turn17view5

The problem is not runtime; the runtime can be quite good at \(O(C A d)\) with \(C\) fixed. The problem is optimization and truncation. DETR-style systems are known to converge slowly enough that major follow-up work was needed to fix training speed, and the fixed-slot budget means that recall failures become hard caps rather than soft ranking errors. In your setting, where missing a hidden tactical pair is often fatal, that is a serious downside. If you do test this family, warm-start it from a dense marginal head and judge it mostly on **recall@C**, duplicate rate, and calibration of the emitted objectness scores rather than on offline likelihood alone. citeturn18view2turn16search0

A learned **generative sampler for progressive widening** is better thought of as a search wrapper than as a standalone policy architecture. Sampled MuZero shows that sampling action subsets can be principled, Gumbel planning gives a compelling way to admit root actions without replacement, and generative slate methods show that diversity-improving generators can help in combinatorial action spaces. This suggests a sensible design: let one of the earlier architectures define a proposal distribution \(q_\theta(\{i,j\}\mid s)\); at the root, admit a without-replacement set using Gumbel noise; deeper in the tree, widen children only when \(N(s)\) grows and sample fresh pairs from \(q_\theta\). Search then works with a moving sampled subset rather than an upfront full candidate set. citeturn13search0turn17view1turn22view0turn25view0turn29view0

I do not recommend starting there. Generative samplers are excellent at hiding selector failure behind stochasticity. They are harder to calibrate, harder to debug, and more vulnerable to early self-play collapse unless you explicitly preserve entropy or reward-proportional diversity. The GFlowNet-style slate literature is attractive precisely because it improves diversity, but it also brings additional hyperparameters and a less transparent training signal. That is useful as a second-stage search upgrade, not as the first policy head you should trust. citeturn29view0

Finally, tree-structured action backends are real options, but I would treat them as engineering infrastructure, not as the core modeling idea. Conditional Action Trees and tree-based retrieval models reduce retrieval complexity by decomposing large action spaces into smaller decisions or coarse-to-fine traversals. If ANN over cell embeddings becomes your latency bottleneck, a learned tree index is worth a look. But it creates an additional learned index to maintain under D6 transformations and canonical pair identity, which is exactly the sort of moving part you said you wanted less of, not more. citeturn23view1turn24view0

## Recommendation and staged experiment plan

My recommendation is to build **one clean baseline plus two challengers**. The baseline should be the factorized marginal + sparse symmetric correction architecture, with three non-negotiable add-ons: a deterministic exact tactical override for immediate win/block pairs, a dense cell-marginal auxiliary target derived from the search pair posterior, and candidate-set normalization for the final MCTS priors. The challengers should be the latent mixture-of-soft-pointers model and the symmetrized autoregressive pair policy. Keep the backbone, value head, training pipeline, and tactical override identical across all three.

The experiment plan should be staged, not monolithic. First, run an **offline selector bake-off** on frozen search data. For small states, build an exhaustive or near-exhaustive audit posterior over all legal pairs. For larger states, use a much larger search budget plus the exact tactical oracle as the audit target. Evaluate best-pair recall@\(C\), covered posterior mass, conditional calibration on the candidate set, terminal win/block recall, D6 consistency, duplicate/illegal rate, selector latency p50 and p95, and pair-scoring throughput. This phase should decide whether the architecture is even allowed into self-play.

Second, run **root-only search integration**. Keep deeper-node logic identical and replace only root candidate admission and priors. This is the cleanest way to test whether better pair priors actually help search, rather than merely improving offline recall. Measure nodes per second, root-expansion quality, and equal-wall-clock Elo against Plan A and Plan B.

Third, run **full self-play with progressive widening** for the top two models only. Do not compare fixed simulation counts; compare equal wall-clock budgets. Track robustness early in training: proposal entropy, candidate-set support size, fraction of search mass outside the candidate set, component usage entropy for mixture models, and the survival rate of low-probability exploratory candidates. If a model gets stronger only after long training but collapses early, that is a real production risk, not a cosmetic flaw.

You should also build **targeted stress suites** rather than relying only on average metrics. One suite should contain immediate tactical states with 4-window and 5-window wins and forced blocks. One should contain “hidden synergy” states where the best pair has low endpoint marginals. One should contain highly symmetric states for D6 consistency checks. One should contain late crowded positions where \(A\) is largest. The architecture choice will likely be determined by the hidden-synergy and tactical suites, not by average-case NLL.

My strong prior is that the factorized marginal + sparse correction architecture will win the first serious comparison because it offers the best tradeoff among dense supervision, calibrated candidate-set priors, symmetry safety, and implementation simplicity. If it fails, it will fail in a very specific and diagnosable way: missing low-endpoint-salience but jointly critical pairs. If that happens, promote the latent mixture-of-soft-pointers model. I would only promote the autoregressive model if its conditional completion distributions are demonstrably sharp and its partner retrieval latency stays comfortably below the retrieval+rereank baseline. I would not make the set decoder or generative sampler the primary production candidate unless the first three all fail on equal-wall-clock strength.