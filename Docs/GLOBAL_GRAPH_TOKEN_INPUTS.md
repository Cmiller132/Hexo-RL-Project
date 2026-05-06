# Global Graph Token Inputs

This document focuses only on the input tokens used by the global graph model family. It is based on the current code in `Python/src/hexorl/graph/batch.py` and the model consumers under `Python/src/hexorl/models/`.

## The Short Version

A CNN sees the board as a fixed 2D image. Every cell is always present in a grid.

A global graph model sees the position as a variable-length list of tokens. Each token is a small record saying "this thing matters": a stone, a legal move, a six-cell line window, a tactical cover set, a connected component, or a global summary token. The model also receives a relation table that tells it which tokens are connected or near each other.

So instead of:

```text
33 x 33 board crop -> CNN channels -> 1089 policy logits
```

the graph path is closer to:

```text
history + legal rows + tactics
  -> token list
  -> token features + token types + token coordinates + token relations
  -> graph/attention trunk
  -> one policy logit per legal action row
```

The model does not output a policy over every cell in a 33x33 crop. It outputs policy logits over the legal action rows supplied in the batch.

## Core Tensor Inputs

The token input is carried by `GraphBatch`.

| Field | Meaning |
| --- | --- |
| `token_features` | Float features for every token, shape `(B, T, 48)`. |
| `token_type` | Integer token kind for every token, shape `(B, T)`. |
| `token_qr` | Axial board coordinate attached to each token, shape `(B, T, 2)`. |
| `token_mask` | Which token slots are real instead of padding, shape `(B, T)`. |
| `relation_type` | Integer relation kind for every token pair, shape `(B, T, T)`. |
| `relation_bias` | Distance-style attention bias for every token pair, shape `(B, 1, T, T)`. |
| `legal_token_indices` | For each legal action row, the token index of its `LEGAL` token, shape `(B, A)`. |
| `legal_qr` | The board coordinate for each legal action row, shape `(B, A, 2)`. |
| `legal_mask` | Which legal action rows are real, shape `(B, A)`. |
| `pair_first_indices` | For pair rows, token index of the first move. |
| `pair_second_indices` | For pair rows, token index of the second move. |
| `pair_token_indices` | Optional token index of a materialized `PAIR_ACTION` token. Often `-1`. |

Here `T` is the token count after padding, and `A` is the number of legal action rows after any legal-row cap.

The important distinction:

- A token is an input object the model can read.
- A legal action row is an output candidate that receives a policy logit.
- A legal action row points back to one `LEGAL` token through `legal_token_indices`.

## How Tokens Are Built

The graph batch builder starts from compact game history and derives:

1. The stones currently on the board.
2. The current player.
3. Placements remaining this turn.
4. Legal moves from the Rust engine when available.
5. Tactical sets such as winning moves, forced blocks, open-four cells, open-five cells, and cover cells.
6. Active windows, lines, covers, components, and optional pair rows.

It then creates a variable-length list of tokens. Every token has:

```text
token_type
token_qr
token_features[48]
optional membership set
```

The membership set is not passed directly as a tensor. It is used to build `relation_type` and `relation_bias`.

## Token Types

The current token type enum is:

| Token | Value | What it represents |
| --- | ---: | --- |
| `STATE` | 0 | Whole-position summary anchor. |
| `TURN` | 1 | Current turn / remaining placements context. |
| `PLAYER` | 2 | Current player and opponent identity tokens. |
| `STONE` | 3 | A placed stone from the move history. |
| `LEGAL` | 4 | A legal move candidate. |
| `HOT_CELL` | 5 | A legal/tactical cell worth extra emphasis. |
| `WINDOW6` | 6 | An unblocked six-cell line segment. |
| `LINE` | 7 | A whole touched row/diagonal summary. |
| `COVER_SET` | 8 | A tactical set of cells related to threat coverage. |
| `COMPONENT` | 9 | A connected component of stones. |
| `PAIR_ACTION` | 10 | Optional explicit token for a two-placement pair row. |

## `STATE`

There is one `STATE` token per position. Its coordinate is `(0, 0)`.

Conceptually, this is the model's "read the whole position here" token. After attention layers, the final `STATE` vector is used for global outputs such as value and tactical summaries.

It is not a board cell. It is an aggregation point.

## `TURN`

There is one `TURN` token per position. It carries the same general feature vector format as other tokens, including current player and placements remaining.

This gives the attention trunk a clear token representing whose turn it is and whether the position is in a one-placement or two-placement decision state.

## `PLAYER`

There are two `PLAYER` tokens:

- one for the current player
- one for the opponent

These are identity/context tokens. They are not board cells. They help the trunk separate "mine" from "theirs" in a way that is available globally.

Stone ownership is still carried on each `STONE` token; the player tokens are extra global context.

## `STONE`

A `STONE` token represents one placed stone.

The source comes from the compact move history:

```text
(player, q, r)
```

The token coordinate is the stone coordinate. Important features include:

- owner relative to the current player
- move age
- nearest own stone distance
- nearest opponent stone distance
- nearest any stone distance
- tactical flags if that coordinate is in a tactical set

If no context budget is set, all stones become tokens. If `max_context_tokens` is set, the builder keeps a recent subset of stones as token inputs, while the full stone set is still used to derive legal moves, active windows, tactical sets, and components.

Stone tokens connect to other tokens through relations:

- to `WINDOW6` tokens that contain the stone
- to `COMPONENT` tokens containing the stone
- to recent nearby stones
- to older/newer stones through age-order buckets
- to tokens on the same line, same axis, or nearby distance buckets

## `LEGAL`

A `LEGAL` token represents one legal move candidate.

This is the key bridge between graph input and policy output. For every legal action row, the batch stores:

```text
legal_token_indices[row] = token index of the matching LEGAL token
```

The policy head gathers those final `LEGAL` token vectors and produces one logit per legal row.

So the graph model's policy is:

```text
legal row 0 -> LEGAL token for move A -> policy logit 0
legal row 1 -> LEGAL token for move B -> policy logit 1
legal row 2 -> LEGAL token for move C -> policy logit 2
...
```

This is different from the CNN policy, where output index corresponds to a fixed board-grid location. In the graph model, output index corresponds to a row in the legal action table.

Important `LEGAL` token features include:

- coordinate
- current player
- placements remaining
- nearest stone distances
- whether the move wins now
- whether it is a forced block
- whether it is part of an open-four or open-five pattern
- whether it belongs to a cover set
- how many active six-cell windows include it

Legal rows can be capped by `max_legal_rows`. Required rows for training targets and selected pair rows are preserved before other rows are ranked and capped.

## `HOT_CELL`

A `HOT_CELL` token is extra context for a legal cell that the tactical scan considers important.

A hot cell may be:

- a winning move
- a forced block
- part of an open-four or open-five pattern
- a cover cell

This token does not become a policy output row by itself. The actual policy candidate is still the `LEGAL` token. The `HOT_CELL` token gives the attention trunk another way to represent "this coordinate is tactically special."

It can relate to windows and cover sets similarly to legal tokens.

## `WINDOW6`

A `WINDOW6` token represents one active unblocked six-cell segment along one of the three Hex axes.

This is the closest graph-token equivalent to "pattern recognition over a local line segment." It is not a sliding CNN patch. It is a specific six-cell tactical object.

The builder considers windows like this:

1. Take every interesting coordinate: every stone and every legal move.
2. For each coordinate, look along each of the three Hex axes.
3. For each axis, create every length-6 segment that could contain that coordinate.
4. Discard windows with no stones.
5. Discard blocked windows containing stones from both players.
6. Deduplicate by `(axis, start_coordinate)`.

That means the graph does not create every possible six-cell window on the board. It creates active windows near stones and legal moves, and only keeps windows that still matter tactically because they are not blocked by both players.

One stone can belong to multiple `WINDOW6` tokens. That is intentional. A stone may be part of several possible six-in-a-row threats depending on which segment and axis you consider.

A `WINDOW6` token stores:

- its axis
- the center coordinate of the six-cell segment
- count of player-0 stones in the window
- count of player-1 stones in the window
- number of empty cells
- number of legal cells

It also has a membership set containing its six cells. That membership set creates relations such as:

- `STONE_IN_WINDOW6`
- `LEGAL_IN_WINDOW6`
- `SAME_WINDOW6`
- `LINE_TO_WINDOW6`
- `WINDOW6_TO_COVER_SET`

Conceptually, a `WINDOW6` token says:

```text
"Along this axis, these six cells form an unblocked possible line.
Here is how many stones are already in it, how many spaces remain,
and which legal moves can affect it."
```

## `LINE`

A `LINE` token summarizes an entire touched row/diagonal along one Hex axis.

Hex has three main axes. A cell belongs to one line on each axis. The builder creates line tokens for lines touched by stones or legal moves.

A `LINE` token stores:

- axis
- line id
- own stone count on that line
- opponent stone count on that line
- longest own run on that line
- longest opponent run on that line

The counts and runs are current-player relative. This is different from `WINDOW6`, whose current features store player-0 and player-1 counts.

Conceptually:

- `WINDOW6` asks "is this exact six-cell segment promising?"
- `LINE` asks "what is the broader row/diagonal situation this segment belongs to?"

The `global_line_window_0` model makes especially direct use of these tokens by selecting final `WINDOW6`, `LINE`, and `COVER_SET` vectors and using their average as tactical context for legal actions.

## `COVER_SET`

A `COVER_SET` token represents a group of cells from the tactical oracle.

These are not generic geometric regions. They are tactical groups such as:

- cells that cover threats
- cells involved in forced blocks
- cells involved in immediate wins
- cells associated with open-four or open-five pressure

The token coordinate is the rounded center of the set. Important features include:

- cover size
- cover memberships
- tactical flags inherited from the coordinate/set

The token has a membership set of board cells. That membership creates relations to legal and hot cells inside the cover.

Conceptually, a `COVER_SET` token says:

```text
"These cells are tactically linked. Moves here answer or create a related threat."
```

## `COMPONENT`

A `COMPONENT` token summarizes a connected component of stones.

The builder finds connected groups using Hex neighbor adjacency. The token coordinate is the rounded center of the component.

The main feature is component size. The component membership set creates `SAME_COMPONENT` relations between the component token and its stones.

Conceptually, this gives the model a way to reason about groups of already connected stones without reconstructing every adjacency only from pairwise distances.

## `PAIR_ACTION`

A `PAIR_ACTION` token represents a two-placement move pair, but it is optional.

For two-placement turns, the batch can enumerate unordered pairs of legal moves. Pair rows always store references to the first and second move token indices:

```text
pair_first_indices[row]
pair_second_indices[row]
```

If `materialize_pair_context_tokens` is false, no extra token is created and `pair_token_indices[row] = -1`. The pair heads still work from the first and second token references.

If `materialize_pair_context_tokens` is true, the builder creates a `PAIR_ACTION` token with:

- center coordinate of the two moves
- pair distance
- whether the pair reaches a win
- whether the pair blocks a threat
- cover size

Conceptually:

- pair rows are output candidates for pair policy heads
- optional `PAIR_ACTION` tokens are input context objects

Those are related but not the same thing.

## The Shared Feature Vector

Every token receives a 48-dimensional feature vector. Current code uses indices `0` through `36`; indices `37` through `47` are spare zeros.

| Index | Meaning |
| ---: | --- |
| 0 | Token type normalized. |
| 1 | `q / 64`. |
| 2 | `r / 64`. |
| 3 | `(q + r) / 64`. |
| 4 | Hex distance from origin, normalized. |
| 5 | Current player. |
| 6 | Placements remaining, normalized. |
| 7 | Owner relative to current player. |
| 8 | Move age, normalized. |
| 9 | Axis encoding. |
| 10 | Player-0 count, normalized. |
| 11 | Player-1 count, normalized. |
| 12 | Legal count, normalized. |
| 13 | Stone count, normalized. |
| 14 | Current player's stone count, normalized. |
| 15 | Opponent's stone count, normalized. |
| 16 | Nearest own stone distance. |
| 17 | Nearest opponent stone distance. |
| 18 | Nearest any stone distance. |
| 19 | Window empty count. |
| 20 | Window legal count. |
| 21 | Own line count. |
| 22 | Opponent line count. |
| 23 | Own line longest run. |
| 24 | Opponent line longest run. |
| 25 | Cover size. |
| 26 | Cover memberships. |
| 27 | Component size. |
| 28 | Pair distance. |
| 29 | Pair reaches win. |
| 30 | Pair blocks threat. |
| 31 | Is win now. |
| 32 | Is forced block. |
| 33 | Is open four. |
| 34 | Is open five. |
| 35 | Is cover cell. |
| 36 | Hot window count. |
| 37-47 | Currently unused. |

Not every feature is meaningful for every token type. For example, `pair_distance` matters for `PAIR_ACTION`; `component_size` matters for `COMPONENT`; line run features matter for `LINE`; tactical flags are most meaningful for `LEGAL` and `HOT_CELL`.

## Relations Between Tokens

The model does not only receive isolated token records. It also receives pairwise token relations.

Examples:

| Relation | Meaning |
| --- | --- |
| `STONE_IN_WINDOW6` | A stone belongs to a six-cell window. |
| `LEGAL_IN_WINDOW6` | A legal move belongs to a six-cell window. |
| `SAME_WINDOW6` | Two tokens share at least one active six-cell window. |
| `LINE_TO_WINDOW6` | A line token and window token describe the same axis/line. |
| `LEGAL_IN_COVER_SET` | A legal or hot cell belongs to a tactical cover set. |
| `WINDOW6_TO_COVER_SET` | A window overlaps a tactical cover set. |
| `SAME_COMPONENT` | Tokens belong to the same connected stone component. |
| `AGE_ORDER_BUCKET` | Stone tokens are related by move age order. |
| `RECENT_MOVE_RELATION` | A recent stone is near another token. |
| `LEGAL_TO_PAIR_ACTION` | Optional pair token connects to its legal move tokens. |
| `FIRST_SECOND_PAIR_RELATION` | First and second moves in a pair row are related. |
| `SAME_AXIS` | Tokens are aligned on a Hex axis. |
| `SAME_LINE` | Tokens are on the same line. |
| `DISTANCE_BUCKET` | Tokens are near/far by Hex distance. |
| `DIRECTION_BUCKET` | Directional relation between token coordinates. |

The relation table influences attention. A token can attend differently to another token depending on whether it is in the same window, same component, same tactical cover, or just nearby.

This is the main reason graph tokens can be sparse. The model does not need a dense board image if the token list and relation table already identify the tactically meaningful objects and how they connect.

## How Major Global Graph Models Use The Tokens

All major global graph variants start from the same basic token table:

```text
token_features + token_type + token_qr + token_mask
```

Most also use:

```text
relation_type + relation_bias
```

The model embeds each token by combining:

- projected float features
- token type embedding
- coordinate embedding
- relation-aware attention inside graph blocks

After the trunk, each token has a final vector.

### Shared Policy And Value Pattern

The standard graph pattern is:

```text
STATE final vector -> value / global heads
LEGAL final vectors -> policy_place logits
pair first/second token vectors -> pair heads
```

The policy head does not score every token. It gathers only the token indices listed in `legal_token_indices`.

### `global_graph_option1`, `global_graph_full_0`, `global_graph768_champion`

These are the most direct graph-trunk models.

They use the token list plus relation tables as the main representation of the position. Their policy is built from final `LEGAL` vectors. Their value comes from the final `STATE` vector.

The larger variants mainly change capacity and architectural settings. They do not change what a token means.

### `global_xattn_0`

This variant treats legal moves as action queries and lets them cross-attend to context tokens.

The practical effect:

- `LEGAL` tokens are the actions being scored.
- Non-legal tokens are context: stones, windows, lines, covers, components, state, players, turn.
- Each legal action can ask "which context tokens matter for me?"

This is useful when you want legal action scoring to be explicit and action-centered.

### `global_line_window_0`

This variant gives special downstream treatment to tactical structure tokens.

After the graph trunk, it selects:

- `WINDOW6`
- `LINE`
- `COVER_SET`

It averages their final vectors into tactical context and uses that context to gate/refine legal action vectors.

Conceptually, this model says:

```text
"Legal moves should be scored through the tactical line/window/cover structure around them."
```

### `global_pair_twostage_0`

This variant focuses on pair-action reasoning for two-placement turns.

It uses token references from pair rows:

```text
first move token
second move token
optional pair token
```

The pair heads reason over these references. The pair does not need to be a normal token unless `materialize_pair_context_tokens` is enabled.

Conceptually:

```text
First-stage policy: which first move looks good?
Second-stage/pair policy: which move combination works?
```

The pair structure is built from legal-token references, not from a fixed 2D output grid.

### `global_hybrid_action_0`

This variant still uses graph legal tokens, but it also has action-specific gating from raw token features and may include crop/CNN context depending on configuration.

It is a hybrid because it keeps the graph action-table policy shape while allowing extra local/crop-style information to influence legal move scoring.

## Why These Token Types Exist

Each token type exists because it gives the attention trunk a useful object that would otherwise have to be rediscovered from scratch.

| Token | Why it exists |
| --- | --- |
| `STATE` | Gives the model a single global readout point. |
| `TURN` | Makes turn structure explicit. |
| `PLAYER` | Makes current-player/opponent identity globally available. |
| `STONE` | Represents the actual played position. |
| `LEGAL` | Represents exactly the actions that can be chosen. |
| `HOT_CELL` | Highlights tactically important legal cells. |
| `WINDOW6` | Represents possible six-in-a-row segments directly. |
| `LINE` | Represents broader row/diagonal pressure. |
| `COVER_SET` | Represents tactical threat-answer groups. |
| `COMPONENT` | Represents connected stone groups. |
| `PAIR_ACTION` | Optionally represents a whole two-placement move as context. |

The design is deliberately not one-token-per-board-cell. It is one-token-per-relevant-object.

That is the central mental shift from CNNs:

```text
CNN:
  "Look at every location in a fixed image."

Graph:
  "Look at the important objects and their relationships."
```

