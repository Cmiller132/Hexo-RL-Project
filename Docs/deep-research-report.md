# Hexo-RL Research Recommendation

## Thesis

The best path is to keep Hexo-RL as an exact-simulator, entity["software","AlphaZero","self-play reinforcement learning game-playing algorithm"]-style system, but make the **searched action** an **unordered joint pair** of placements for each full turn. Do **not** make the main tree sequential over “first stone, then second stone” afterstates. Instead, use sequential or autoregressive structure only to **propose** candidate pairs cheaply. The actual tree should reason over the true move object: a two-stone turn. Your attached handoff is right that this is the central design decision, and it is also right to favor a variable-size graph/object representation over a fixed crop. fileciteturn0file0 citeturn1search0turn0search1turn14view4turn17view0turn14view0turn18view0

If I were choosing one architecture today, I would build a **hybrid sampled joint-pair search**: a graph/set encoder over stones, legal cells, and six-cell threat windows; a cheap autoregressive single-cell proposal module; a symmetric joint pair scorer; **Gumbel-style root candidate admission**; and **PUCT with progressive widening** over admitted joint-pair children deeper in the tree. The key change to your current plan is not “abandon pair actions.” It is “**keep pair actions as first-class, but never enumerate them exhaustively, and never train the pair head from raw sparse visit counts alone**.” citeturn18view0turn17view0turn26view3turn26view5turn24view0

I would also reject a learned-dynamics entity["software","MuZero","model-based reinforcement learning algorithm"] mainline. MuZero’s contribution is learning a planning model when the environment dynamics are unknown or not given directly. Hexo has cheap exact rules. Your bottleneck is not transition modeling; it is **combinatorial branching over pair actions**. The relevant MuZero-era papers for Hexo are the ones about **sampled action subsets** and **planning targets**, not learned dynamics. citeturn0search1turn17view0turn16view4

## Recommended search architecture

### Why pair actions should be first-class

In Hexo, most of the strategically decisive moves are not “good first stones that happen to be followed by decent second stones.” They are **synergistic pairs**: double extensions, two-cell blocks, overload creation, one-cell tempo plus one-cell cover, or multi-axis pressure that only exists as a pair. That is exactly the kind of interaction that factored-action methods struggle with when the reward comes from **coupling between sub-actions** rather than from largely independent contributions. Classical work on entity["other","Connect6","k-in-a-row game with two stones per turn"] already emphasized threat-based search because the game’s strategic core lives in threat interactions, not in independent single placements. citeturn21view1turn20search1turn30search5turn2search3

A sequential afterstate tree is therefore the wrong primary semantics. It introduces an **artificial order** into an unordered pair, distorts credit assignment toward the first placement, and teaches the search to optimize an intermediate object that the real game never asks you to choose. Afterstates are useful in RL when an action naturally decomposes into “act, then environment/opponent responds,” but here the environment does **not** intervene between the two placements. The afterstate concept itself is sound; it is just not the correct top-level decision object for Hexo’s full turn. citeturn9search6turn8search11

### The exact search recipe

At each full-turn node, I would do the following.

First, encode the current position once and produce embeddings for occupied cells, legal cells, and six-cell windows. Then generate a **candidate pair set** from multiple sources rather than a single policy prior. The sources should be: a policy-driven autoregressive proposal, a tactical exact-rule proposal, and a structured exploration proposal. This is the Sampled MuZero lesson adapted to a known simulator: search only a sampled subset, but do it in a proposal-aware way and keep the proposal probabilities. citeturn17view0turn16view4turn17view3

Second, deduplicate every pair into a canonical unordered representation before the tree sees it. The tree child should be `(a,b)` with `a < b` under a fixed canonical ordering after symmetry normalization, never `(a then b)` versus `(b then a)`. This is not a cosmetic choice. It removes gratuitous aliasing and makes the search statistics correspond to the real move space. Your handoff is right to treat this as a correctness issue. fileciteturn0file0

Third, at the **root**, do **Gumbel candidate admission / sequential halving** over the sampled pair set instead of raw PUCT over everything. Gumbel planning was built precisely to keep policy improvement working under small simulation budgets, and it empirically improves low-simulation learning relative to standard MuZero and AlphaZero. In 9x9 Go, Gumbel MuZero still learned with tiny simulation budgets where plain MuZero did not, and its root search explicitly uses sampled actions without replacement and completed-Q policy targets rather than treating root search as a bag of heuristic counts. That applies directly to Hexo because you will also be acting from **small, sampled action subsets**. citeturn14view0turn18view0

Fourth, at **non-root nodes**, I would use standard PUCT over the currently admitted pair children, but attach **progressive widening** so highly visited nodes can request more pair candidates later. Sampled MuZero explicitly points out that if simulations are much larger than the initial sample size, progressive widening is the natural extension. This is a better fit for Hexo than trying to decide a giant candidate set upfront for every interior node. citeturn17view0turn6search0turn6search1

Fifth, use **policy target pruning** for exploratory pair candidates. If you force visits to noisy, tactical, or low-prior pairs to ensure recall, do not directly train on the raw resulting visit distribution. KataGo’s forced playouts and policy target pruning are the closest existing solution to exactly your concern: exploration is necessary to discover underestimated actions, but the raw explored distribution is a polluted policy target. The same principle should be applied to Hexo pair candidates. entity["software","KataGo","open-source Go engine"] showed that decoupling exploration from the learned policy target is one of the highest-leverage improvements over vanilla AlphaZero. citeturn26view3turn26view5

### Why the alternatives lose

A **pure sequential-afterstate tree** is the strongest competing baseline for wall-clock, but I would still reject it as the mainline. If you test it, test the **best possible** version: merge transpositions in a DAG, because the sequential formulation creates many order aliases and repeated substructures. Monte-Carlo Graph Search exists precisely because DAG search can share information across transpositions and improve both fixed-time and fixed-evaluation performance. If your sequential baseline is only a tree, it is not a fair comparison. Even then, I expect it to lose in final strength because it reasons over the wrong action semantics. citeturn27view0

A **pure factorized-action MCTS** is also the wrong mainline. Factored MCTS and newer action-abstraction work are useful when sub-actions are only moderately coupled and the search mostly suffers from irrelevant combinations. Hexo is the opposite. The value of a pair often comes from exact combinational interactions: two hot windows covered by one pair, a fork across two axes, or a pair that is worthless unless both cells are played together. Those papers are relevant as evidence that large factored action spaces need structure, but they do **not** imply that Hexo should back up independent single-cell decisions as if the pair were approximately separable. citeturn23view1turn24view0

A **MuZero-style sampled subset over latent dynamics** is unnecessary overhead. The useful part is the sampled-subset policy-iteration formalism, not the learned model. And a **pure autoregressive pair chooser** is too weak as the final scorer. Autoregressive decoders are excellent proposal mechanisms for variable candidate sets, but in Hexo they should feed a joint pair reranker, not replace it. Pointer-style decoders are relevant because the output dictionary is variable-sized, while biaffine relation scoring is relevant because the final decision is a relation between two legal cells. citeturn14view8turn10search2turn17view0

### Fixed-simulation strength and fixed-wall-clock strength

At a **fixed number of simulations**, the sampled joint-pair tree should be stronger than any sequential-afterstate variant, because every search edge corresponds to the actual move object and therefore every backup targets the right policy. Gumbel-style root planning should widen that gap when simulation budgets are still small. citeturn18view0turn14view4

At a **fixed wall-clock budget**, a naive joint-pair tree would lose badly, but the proposed hybrid should still be the best long-run design because the expensive part is the encoder, not the pair scoring: once you have cell embeddings, batched pair scoring is cheap relative to additional node expansions. The sequential baseline may look better during very early training if your proposal recall is poor. That is why you should benchmark against a DAG afterstate baseline, but I would still optimize the joint-pair search rather than switch semantics. citeturn27view0turn24view0

## Recommended model architecture

### The encoder

Use a **graph/set transformer over objects**, not a CNN crop and not pair tokens in the main attention stream. The relevant prior here is clear: graph-based AlphaZero variants scale better across board sizes and variable board structures, and Set Transformer-style attention is explicitly designed for variable-cardinality sets where interactions among elements matter. Hexo is even more naturally object-centric than Go or Othello because the board is unbounded and the legal region is sparse. citeturn29view0turn29view1turn14view6

The encoder should contain four object types: occupied stone cells, legal move cells, six-cell line windows, and a small number of global state tokens. The six-cell window tokens are not optional decoration. They are the shortest exact objects that express “four with two empties,” “five with one empty,” overlap structure, and pair-cover motifs. Connect6 research repeatedly returns to threat windows for a reason: in two-stone connection games, threat geometry is the natural latent space. fileciteturn0file0 citeturn21view1turn20search1turn14view12

I would connect cells to windows by incidence edges, cells to nearby cells by relative axial offsets, and windows to windows when they overlap or share cover cells. That lets the net represent both local shape and overload structure. Use relative-position encodings and D6 symmetry augmentation, but do not spend your compute budget trying to materialize all possible pair tokens globally. Your handoff is correct to reject that. fileciteturn0file0

### The action heads

The best model is **three-headed**, not one-headed.

The first head is a **single-cell proposal head** over legal cells. Its job is not to define the final move. Its job is to cheaply identify promising anchors, coverage cells, and rare tactical cells. This head should also predict dense tactical cell labels, because that gives it much more signal than sparse pair outcomes alone. citeturn25view0turn26view3

The second head is an **autoregressive conditional head** for a second cell given the first. This is the Hexo use-case where Pointer Network logic is actually valuable: it handles a variable-size legal dictionary and is extremely efficient for proposal generation. But I would use it for **candidate generation**, not for the final policy semantics. citeturn14view8

The third head is the important one: a **symmetric joint pair reranker**. This should score unordered pairs directly using the state embedding, both cell embeddings, and explicit pairwise geometry/tactics. A low-rank biaffine term is the right backbone because the action is fundamentally a relation between two chosen cells. The final pair score should look like “biaffine relation score plus exact pair features plus global state context,” not merely an MLP over summed embeddings. Biaffine scoring is the closest mature template for this kind of pair relation. citeturn10search2turn10search5

In practice I would compute something like  
\(s_{ij} = h_i^\top U h_j + w^\top [h_i, h_j, |h_i-h_j|, h_i \odot h_j, g_{ij}, z]\),  
where \(g_{ij}\) contains explicit geometric and tactical pair features: axial distance, same-axis indicator, number of self hot windows completed, opponent hot windows covered, number of overlapping cover sets, and overload/fork counts. That gives you the expressive joint scorer you need without turning all pairs into first-class tokens.

### What not to do

Do **not** make the pair head your only head. Pure sparse pair supervision will undertrain the representation. Do **not** rely on a symmetric pair MLP alone if you can afford a biaffine relation term; the weaker head will miss subtle pair-specific complementarity. And do **not** let the autoregressive factorization define the tree semantics simply because it is easier to code. That convenience cost shows up later as search bias. citeturn10search2turn14view8turn17view0

## Exploration and training

### How exploration should work

The exploration floor should live in **candidate-source quotas**, not in a large direct **uniform-over-all-pairs** distribution. This is one of the weakest assumptions in your current handoff. A raw uniform pair proposal over a huge quadratic action space spends almost all of its mass on strategically void pairs. A much better exploration scheme is: some policy-driven anchors, some exact tactical proposals, and some structured exploratory proposals such as “uniform first cell, diverse second cell” or “uniform over windows / covers / distances,” all with nonzero quotas. That keeps recall without drowning the tree in meaningless pairs. fileciteturn0file0 citeturn17view0turn26view5

Very early in training, I would **not** start from a truly random pair policy. I would pretrain on synthetic exact tactical labels generated from the rules: immediate wins, forced blocks, hot-window counts, pair-cover counts, overload candidates, and short-horizon tactical outcomes. Classical Connect6 programs used threat-space machinery because these signals matter, and KataGo showed that auxiliary supervision and search/training decoupling dramatically accelerate self-play learning. Hexo should exploit both lessons. citeturn21view1turn14view12turn14view3turn26view3

Then use a **two-budget self-play schedule** similar in spirit to KataGo’s playout cap randomization: many positions searched cheaply for value/data throughput, and a smaller fraction searched deeply enough to produce strong policy-improvement targets. This directly addresses the same policy/value tension that KataGo identified in AlphaZero training. citeturn32view0

As the policy matures, anneal the exploration mixture **source-wise**, not by pretending the whole pair space has become tractable. Reduce the structured exploratory quota, keep a persistent tactical quota, and continue to inject candidate novelty at the root. Also add **policy surprise replay weighting** so states where search sharply disagrees with the prior are replayed more often. That is a particularly good fit for Hexo because rare overload and multi-axis motifs will otherwise remain underrepresented. citeturn25view2

Finally, use **targeted search control** from archived hard states rather than always self-playing from the empty start state. Go-Exploit showed that AlphaZero underexplores deeper states when it always starts from the root and only samples actions in the opening. Hexo’s decisive tactical states are exactly the kind of deep, low-frequency configurations that benefit from this correction. citeturn13search7turn14view11

### Whether sparse joint pair targets can train well

Yes, **but not by themselves**.

Raw sparse joint-pair visit targets are too brittle for Hexo. They will be high variance, candidate-set-biased, and prone to collapse because the searched support is tiny relative to the legal pair space. The right question is not “can sparse pair targets work?” It is “what **additional views** have to be trained alongside them?” The answer is: at least four. citeturn17view0turn26view3turn25view0

The first required view is a **pruned candidate-set pair posterior**. Use the searched candidate set only, prune purely forced exploratory traffic, and preferably convert search statistics into a **completed-Q / regularized posterior** rather than literal raw counts when visits are sparse. That follows the logic of Gumbel planning and regularized policy optimization, which treat search as a posterior policy-improvement operator rather than as a bare histogram. citeturn14view4turn18view0

The second required view is the **marginal cell target**. Every joint-pair target should be projected into a per-cell probability target so the cell head learns which cells commonly participate in strong pairs even if the exact pair itself was rarely sampled. The third required view is a **conditional second-cell target** for the searched first cells. The autoregressive proposal learns pair structure much faster from this than from sparse joint-labeling alone. The fourth required view is a **softened pair target**, not just the hard target. KataGo’s later auxiliary soft policy target is the closest precedent: lower-mass but meaningful actions need gradient too, or the network learns only the top one or two choices. citeturn25view0

The rule I would enforce is simple: **never treat unsampled legal pairs as hard negatives**. If the tree never saw them, they are “unknown,” not “bad.” Only candidate pairs that were admitted and judged weak by search are trustworthy negatives. If you want to approximate a global pair softmax over sampled negatives, store the proposal probability \(\beta\) for every sampled pair and use a proposal-corrected sampled loss. Sampled MuZero gives the policy-iteration logic for sampled subsets, and sampled-softmax literature shows that naive sampled classification is biased when proposal probabilities are ignored. citeturn16view4turn17view0turn28academia15turn28academia16

## Learning Hexo-specific tactics

Hexo-specific tactics should be learned by **exact local supervision plus search**, not by hoping the value head rediscovers them from win/loss alone. Four- and five-stone windows, pair blocks, overload, and multi-axis pressure are not optional heuristics. They are the game’s native local invariants. Connect6 literature and engines treat threat windows as first-class because they are the shortest exact summary of tactical inevitability. citeturn21view1turn20search1turn2search3

I would create exact auxiliary labels for every hot six-cell window and every legal cell: whether the window is self-live or opponent-live, whether a cell covers a hot opponent window, how many hot windows it participates in, whether it creates a self immediate win next turn, and whether it contributes to a minimum pair cover of opponent threats. This is the right place to inject “domain knowledge”: as exact labels and proposal generators, not as a hand-authored value function. KataGo’s efficiency gains from auxiliary targets and Connect6’s historical reliance on threat-space search both point the same way. citeturn14view3turn14view12turn26view3

I would also add **short-horizon tactical heads**. For example: “win in one full turn,” “must block opponent hot threats this turn,” “no legal pair covers all opponent immediate threats,” and “number of independent self pressure axes.” In Go, KataGo benefited from short-term targets and richer auxiliary supervision because they regularized the representation and improved later search usefulness. Hexo’s tactical motifs are even more local and exact than Go territory/score structure, so I expect these heads to matter more, not less. citeturn25view0turn32view0

The one hard-coded solver I would permit in the mainline is a **near-terminal tactical oracle** for exact one-turn wins, exact impossible-to-cover positions, and bounded-depth threat races on tiny hot-window subgraphs. That is not “cheating.” It is the board-game analogue of a terminal solver, and MCGS-style work as well as solver papers in Gomoku/Connect6 both support the value of exact terminal reasoning once the search enters a forcing regime. citeturn27view0turn31view0

## Weak assumptions to correct

The weakest assumptions in the attached plan are these.

- **A persistent uniform-over-pairs exploration floor is a good long-run idea.** I think this is wrong. Keep persistent exploration, yes, but mostly as **source quotas** over first cells, tactical proposals, and structured novelty. Pair-uniform mass becomes vanishingly ineffective as the legal pair space expands. fileciteturn0file0 citeturn17view0turn26view5

- **Raw sparse joint pair MCTS targets are probably enough if the model is strong enough.** I think this is wrong. They need pruning, softening, candidate-set awareness, and dense projected auxiliaries. Otherwise you will get self-confirming support collapse. citeturn18view0turn26view3turn25view0

- **A simple pair MLP over summed/absolute-difference/product features is enough as the main pair scorer.** I think this is too weak. Use a true relation scorer, ideally biaffine plus explicit pairwise tactical features. citeturn10search2turn10search5

- **A sequential-afterstate tree is a plausible primary search semantics if pair branching is too large.** I think this is wrong as the mainline. It is a good ablation baseline, but only if you add DAG transposition handling; otherwise you are testing an artificially bad version of the alternative. citeturn27view0turn9search6

- **MuZero-style learned dynamics are an attractive option because the action space is hard.** I think this is wrong. The action space is the problem; the state transition is not. Use exact rules and spend your complexity budget on search control and pair scoring. citeturn0search1turn17view0

- **Curricula that alter the action semantics are probably fine.** I would be careful. If you use curriculum at all, preserve the two-placement full-turn semantics and the same tactical objects. Mixed-size or masked-size training can help when local rules are preserved, but I would not pretrain on a one-stone-per-turn surrogate game and expect clean transfer. citeturn29view0turn29view1

## Ablations and future exploration

The first ablations I would run are not generic hyperparameter sweeps. They should answer the architectural question decisively.

- **Search-family shootout.** Compare three systems with the same encoder budget: sampled joint-pair search; sequential afterstate DAG search; and a hybrid that proposes sequentially but searches joint pairs. Evaluate at equal wall-clock, equal simulations, and equal neural evaluations. If you do not run all three, you will not know whether your main gains came from semantics or from throughput. citeturn27view0turn17view0turn18view0

- **Pair-head ablation.** Compare autoregressive-only, symmetric MLP, and biaffine joint reranker. The metric is not just Elo. It is candidate recall of the eventual search-selected pair, recall of exact tactical winning pairs, and KL from search posterior to model posterior on candidate sets. citeturn10search2turn14view8

- **Target-stack ablation.** Compare raw visit-count pair CE against: pruned pair target; completed-Q posterior target; plus marginal-cell auxiliary; plus conditional-second auxiliary; plus softened pair auxiliary. I expect this ablation to be one of the most decisive in the whole project. citeturn18view0turn26view3turn25view0

- **Candidate-source ablation.** Policy only versus policy+tactical versus policy+tactical+structured exploration. The key metric is “best-search pair in candidate set?” and “exact block/win pair in candidate set?” not just game outcomes. citeturn17view0turn26view5

- **Tactical-object ablation.** Remove six-cell window tokens, remove exact tactical auxiliaries, or remove pairwise tactical features. If those removals do not hurt, your model is not actually using Hexo-specific structure and you are leaving strength on the table. citeturn21view1turn14view12

- **Search-control ablation.** Plain self-play from the empty board versus archived hard-state starts and surprise-weighted replay. If Go-Exploit-like state control does not help, the tactical labels or proposal recall probably still need work. citeturn13search7turn14view11turn25view2

What success should look like is also clear. The joint-pair architecture should beat the sequential DAG baseline at equal wall-clock after a modest amount of training, not just at equal simulations. The candidate set should contain the final best-search pair on an overwhelming majority of moves. Exact one-turn winning and must-block pairs should have near-perfect recall. And policy collapse should be visible if it happens: falling candidate-source diversity, low entropy too early, and large persistent surprise KL between search and prior. citeturn18view0turn25view2turn26view3

For future exploration, the most interesting alternatives are: a stronger pair-conditioned reranker over only the top few dozen root candidates; a bounded-depth exact threat oracle for hot-window subgraphs; and newer combinatorial-action policy families such as set-based or diffusion-style proposal models. I would treat those as **proposal upgrades**, not as replacements for the sampled joint-pair tree, unless the mainline loses the fair wall-clock search-family ablation above. citeturn24view0turn31view0

My bottom-line recommendation is therefore decisive: **search joint pair actions, propose them with autoregressive singles, rerank them with a symmetric pair head, prune exploration from training targets, and teach the model Hexo’s tactical objects explicitly through exact auxiliary supervision.** That path is the best match to the game’s true move semantics, the strongest route at fixed simulations, and the most likely best route at fixed wall-clock once the proposal model stops being weak.