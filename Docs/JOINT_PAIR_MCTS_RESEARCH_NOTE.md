# Joint Pair MCTS Research Note

## Hypothesis

The strongest research-backed architecture for Hexo is to treat a full
two-placement turn as one first-class MCTS action: `(q1, r1, q2, r2)`. The
search edge is the complete pair, the model prior is a distribution over legal
pair actions, expansion applies both stones atomically, and training uses MCTS
visit counts over joint pairs as the primary policy target. Single-placement
heads can remain useful as proposal, auxiliary, or diagnostic heads, but they
must not be the semantic authority for search. The reason is simple: a pair can
be valuable because of the interaction between its two stones, and marginalizing
back to independent placements destroys exactly the thing the pair head is
supposed to learn.

This design follows AlphaGo Zero and AlphaZero at the highest level: a neural
policy/value model guides MCTS, MCTS produces an improved search policy, and
self-play trains the next model from that improved policy. The difference is
that Hexo's action space is combinatorial. If there are `m` legal placements,
there are `m * (m - 1) / 2` unordered legal pair actions. That makes plain
AlphaZero-style full action enumeration a poor fit. The best version is
therefore pair-action MCTS with sampled candidate admission, Gumbel sampling
without replacement, progressive widening, and an explicit exploration floor.
This combines the neural-MCTS policy-improvement idea from AlphaZero, the
sampled-action framing from Sampled MuZero, the few-simulation candidate
selection idea from Gumbel AlphaZero/MuZero, and the large-action-space control
provided by progressive widening and factored-action MCTS.

## MCTS Design

Each decision node defines the legal pair action set:

```text
A(s) = {(i, j): i and j are legal placements, i < j}
```

Children are keyed by canonical pair identity, not by first stone followed by
second stone. A simulation selects among expanded pair children with a PUCT-style
score:

```text
score(s, a) = Q(s, a)
            + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

When a pair child is expanded, both placements are applied as one macro-action
and the resulting state is evaluated. Backpropagation updates the pair edge.
If the game engine internally creates afterstates after the first placement,
those afterstates should be implementation details of the pair transition, not
independent policy decision roots.

The key is that top-k must never be a permanent gate. Fixed top-k pair sampling
can miss a winning pair forever if the early model does not rank it highly. The
search should instead use progressive widening:

```text
allowed_children(s) = min(|A(s)|, ceil(c_pw * N(s)^alpha_pw))
```

with `0 < alpha_pw < 1`, commonly around `0.4` to `0.6` as an initial research
range. If a node has fewer expanded pair children than `allowed_children(s)`, a
new pair is admitted from a mixed proposal distribution. Otherwise selection
continues among expanded pair children. This means MCTS can start with a small
set of promising pairs, but high-visit nodes keep admitting new pairs. Strong
pairs outside the initial top-k remain discoverable because every legal pair has
nonzero admission probability through the uniform tail and through widening over
time.

The ideal pair proposer is mixed:

```text
beta(a | s) =
  lambda_policy   * joint_pair_policy(a | s)
+ lambda_uniform  * uniform_legal_pairs(a | s)
+ lambda_tactical * tactical_pair_sampler(a | s)
+ lambda_factored * factored_pair_sampler(a | s)
```

The factored sampler may use single-placement priors, marginal Q statistics,
conditional estimates, or pair residuals to propose candidates, but the selected
tree edge remains the full pair. This gives the system the scalability benefit
of factorization without weakening the game semantics into two independent
moves.

## Exploration

PUCT is the right selection rule once actions are admitted, but it does not solve
candidate starvation by itself. Dirichlet noise also does not solve candidate
starvation if the candidate set was produced by deterministic top-k. It only
perturbs actions already present. Gumbel sampling is useful because it samples
without replacement and is explicitly motivated by policy improvement when not
all root actions are visited. But Gumbel should be used as part of candidate
admission, not as blind trust in an immature policy.

Early training should assume the joint pair policy is random or poorly
calibrated. The proposal should therefore be exploration-heavy:

```text
early:
  high lambda_uniform
  high-temperature policy samples
  tactical and local-pattern pair samples
  pairs from single-placement cross-products
  large root candidate set
  aggressive progressive widening
  root noise and high visit temperature

middle:
  lower lambda_uniform
  more weight on joint pair policy
  maintain tactical/factored admission
  shrink temperature only after entropy and calibration improve

late:
  mostly model-guided Gumbel/sample admission
  small but nonzero uniform tail
  lower visit temperature for strength
  occasional stochastic roots to protect against blind spots
```

The durable rule is: `lambda_uniform` should anneal but not vanish. If it reaches
zero, the policy can create self-confirming blind spots. Candidate-set logs
should record the proposal source for each admitted pair so training and
diagnostics can distinguish policy-selected, uniform, tactical, and factored
actions.

If training uses sampled action subsets, the proposal distribution should be
logged. Sampled MuZero's lesson is that search and learning should account for
the difference between the target policy and the action proposal used to form the
candidate set. For Hexo, that means storing the candidate set, proposal weights,
legal-pair count, visit counts, and any normalization metadata needed to train a
correct sparse target rather than pretending the sampled set was the whole
action space.

## Training Feasibility

Pair MCTS should be trainable, but it is higher variance and more fragile than a
single-placement policy. The pair policy target should be the search-improved
visit distribution over legal pair actions:

```text
pi_pair(a) = N(s, a)^(1 / tau) / sum_b N(s, b)^(1 / tau)
```

For large action spaces, this target should be stored sparsely: visited pairs,
candidate pairs, visit counts, proposal metadata, and total legal-pair count.
Dense vectors over all pairs will be mostly zeros and expensive. Auxiliary
single-placement targets can be derived by marginalizing the joint target:

```text
pi_cell(i) = sum_j pi_pair(i, j)
```

but this should be an auxiliary loss only. The main policy loss must train the
joint pair head from the distribution that MCTS actually controls.

The main training risks are search starvation, entropy collapse, calibration
drift, replay skew, and credit assignment blur. Search starvation happens when
too few legal pairs are ever admitted. Entropy collapse happens when the pair
head becomes overconfident before the value model is reliable. Calibration drift
matters because a badly calibrated pair prior can steer both candidate admission
and PUCT in the wrong direction. Replay skew is likely because opening positions
have enormous pair spaces while late-game tactical pair choices may be rarer but
more important. Credit assignment is naturally harder because the value target
says whether the pair worked, not which stone inside the pair was responsible.

The training plan should therefore include a cautious bootstrap: exploration
heavy self-play, tactical pair suites, small-board or reduced-legal curricula
only as temporary support, balanced replay by game phase and legal-pair count,
and direct calibration checks on pair logits. Warm-starting from a
single-placement model may help the trunk, but the final policy must be trained
as a pair policy. The architecture should not let independent placement heads
define the action chosen by MCTS.

## Required Diagnostics

The minimum research dashboard should track:

```text
legal pair count per root
admitted pair count per root
pair candidate source mix
normalized pair target entropy H(pi_pair) / log(|A|)
top-k visit mass
legal-pair coverage over time
visit-count Gini
KL(network pair prior || MCTS pair target)
policy calibration / ECE by visit and outcome bucket
root Q variance
value calibration by game phase
pair replay distribution by opening, midgame, endgame
fixed-simulation Elo
fixed-wall-clock Elo
tactical pair-suite accuracy
```

The acceptance standard should be strength per wall-clock and stable self-play
improvement, not just fixed-simulation win rate. A pair-action tree that wins at
equal simulations but loses at equal wall-clock may still be the wrong practical
architecture.

## Research Basis

- Silver et al., "Mastering the game of Go without human knowledge"  
  https://www.nature.com/articles/nature24270
- Silver et al., "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm"  
  https://arxiv.org/abs/1712.01815
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with
  a Learned Model"  
  https://www.nature.com/articles/s41586-020-03051-4
- Danihelka et al., "Policy improvement by planning with Gumbel"  
  https://openreview.net/forum?id=bERaNdoegnO
- Hubert et al., "Learning and Planning in Complex Action Spaces"  
  https://proceedings.mlr.press/v139/hubert21a.html
- Chaslot et al., "Progressive Strategies for Monte-Carlo Tree Search"  
  https://doi.org/10.1142/S1793005708001094
- Couetoux et al., "Continuous Upper Confidence Trees"  
  https://doi.org/10.1007/978-3-642-25566-3_32
- Kwak et al., "Efficient Monte Carlo Tree Search via On-the-Fly
  State-Conditioned Action Abstraction"  
  https://proceedings.mlr.press/v244/kwak24a.html
- Moraes et al., "Action Abstractions for Combinatorial Multi-Armed Bandit Tree
  Search"  
  https://webdocs.cs.ualberta.ca/~santanad/papers/2018/moraesMLN18.pdf
- Hao et al., "Multiagent Gumbel MuZero: Efficient Planning in Combinatorial
  Action Spaces"  
  https://doi.org/10.1609/aaai.v38i11.29121
- Guo et al., "On Calibration of Modern Neural Networks"  
  https://proceedings.mlr.press/v70/guo17a.html
