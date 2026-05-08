# Binned Encoder Pair Model Concept

## Core Idea

Each legal stone receives a learned latent code. A lightweight decoder compares
or combines two stone codes to produce a pair score. The highest-scoring pairs
become candidate actions for pair MCTS.

```text
board state
  -> legal-cell encoder
  -> per-stone latent code
  -> pair-code decoder
  -> ranked pair candidates
  -> pair MCTS
```

The "bins" are not human-defined categories. They are an emergent internal
language learned by the model because useful pair actions become easier to
retrieve.

## Minimal Shape

For each legal cell:

```text
cell_marginal_logit
pair_code[32-128]
```

For each possible pair:

```text
pair_score(i, j) =
  marginal_i
  + marginal_j
  + decode(pair_code_i, pair_code_j)
  + optional_geometry_bias(i, j)
```

The decoder could be similarity, query-key compatibility, or a learned
compatibility matrix. The important part is not the exact formula; it is that
the model learns a compact code that makes strong pairs easy to find.

## Goal

The goal is learned candidate admission.

Instead of hand-building most pair candidates or running a heavy scorer over
many explicit pair rows, the model would expose a fast pair-retrieval space:

```text
legal stones -> latent codes -> top-K decoded pairs
```

MCTS would then search those candidates, with tactical and diversity sources
still added as safety nets.

## Desired Emergent Properties

The code should not mimic supervised human labels like attack, defense, cover,
axis, or completion. It should invent whatever structure helps pair search.

Desired behavior:

- strong pair actions decode to high scores,
- similar pair roles may cluster when useful,
- complementary pair roles can also score highly,
- the model can represent multiple kinds of pair relationships,
- candidate retrieval improves as policy/search targets improve,
- tactical and diversity sources protect early training while the code is weak.

The model should learn a private pair-compatibility language that lets MCTS
see hundreds of plausible pair candidates without enumerating every pair
through an expensive pair head.

