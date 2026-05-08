# Global Graph Model Explainer

## Audience

This note assumes you already understand the standard Hexo CNN path:

- the board is encoded as a fixed crop tensor, currently shaped like
  `(B, 13, 33, 33)`;
- convolutions build spatial features over that crop;
- the policy head returns logits over fixed board indices;
- the value head returns a binned value distribution;
- optional heads add sparse policy, pair policy, lookahead, regret, axis, or
  other auxiliary signals.

It also assumes a general transformer mental model:

- tokens are embedded into vectors;
- attention lets each token read from other tokens;
- masks remove padded or invalid tokens;
- relation or positional information can bias attention.

The global graph path keeps the transformer-style token idea, but changes what
the tokens mean and, more importantly, what a policy row means.

## Short Version

The standard CNN asks:

```text
Given a fixed spatial crop, score every board cell or selected candidate cell.
```

A global graph model asks:

```text
Given a compact graph of the current state, score exactly the legal action rows
that Rust/MCTS will consume.
```

That sounds small, but it changes the correctness boundary.

The CNN policy is tied to a fixed board coordinate grid. A global graph policy
is tied to a legal row table. The row table is not incidental metadata; it is
the identity of the logits. A logit at column 17 means "the 17th legal row in
this exact legal table", not "board index 17".

## The CNN-To-Graph Translation

Start with the CNN model, because that is the cleanest bridge.

In the CNN path, the model output is easy to picture:

```text
policy logits: shape (B, 1089)
value logits:  shape (B, 65)
```

The policy columns have a permanent meaning:

```text
column 0    -> fixed crop/board cell 0
column 1    -> fixed crop/board cell 1
...
column 1088 -> fixed crop/board cell 1088
```

So the policy tensor itself is almost enough to interpret the output. You still
need legality masks for search, but the column identity is built into the fixed
grid.

In the global graph path, the model still outputs tensors. It does not output
"tokens" as the policy. The common search outputs are:

```text
policy_place logits: shape (B, A)
value logits:        shape (B, 65)
```

The difference is that `A` is not 1089.

`B` is the batch size. If the model evaluates 32 positions at once, `B = 32`.

`A` is the action-table width for that batch. More concretely:

```text
A = the number of legal action entries after batching/padding
```

For one position, if there are 4 legal moves, then that position needs 4 policy
logits. For a batch, positions can have different legal counts, so the collator
pads them to the largest legal count in the batch and carries `legal_mask` to
mark which entries are real.

Each `policy_place` column is one slot in that legal action table:

```text
column i -> legal_qr[i]
```

Here, "column" just means "the i-th number in the policy vector." In the CNN,
the i-th number maps to a fixed board index. In the graph model, the i-th number
maps to the i-th row of `legal_qr`.

Example:

```text
legal_qr = [
  (0, 0),
  (1, -1),
  (3, 2),
  (-2, 4),
]

policy_place logits = [
  0.2,
  1.7,
 -0.4,
  0.9,
]
```

This means:

```text
column 0/logit 0.2  is for row 0: move (0, 0)
column 1/logit 1.7  is for row 1: move (1, -1)
column 2/logit -0.4 is for row 2: move (3, 2)
column 3/logit 0.9  is for row 3: move (-2, 4)
```

The model did not output coordinates. The graph batch already had the
coordinates in `legal_qr`. The model output is a vector of scores in the same
order as that table.

That is what a row is: one entry in a semantic table that gives a model-output
column its meaning.

## Minimal Vocabulary

These words are easy to blur together, so here is the strict meaning in this
codebase.

### Action

An action is the game move search can take, usually represented by a coordinate:

```text
(q, r)
```

In pair contexts, an action row can represent two coordinates:

```text
(first_q, first_r, second_q, second_r)
```

### Row

A row is one entry in a table.

For legal policy, a row is one legal move:

```text
row 0 = (0, 0)
row 1 = (1, -1)
row 2 = (3, 2)
```

For pair policy, a row is one pair:

```text
row 0 = ((0, 0), (1, -1))
row 1 = ((0, 0), (3, 2))
```

Rows are not neural-network vectors. Rows are the semantic labels for output
columns.

### Column

A column is one position in an output tensor.

If:

```text
policy_place = [0.2, 1.7, -0.4]
legal_qr     = [(0, 0), (1, -1), (3, 2)]
```

then:

```text
column 0 means row 0 means action (0, 0)
column 1 means row 1 means action (1, -1)
column 2 means row 2 means action (3, 2)
```

### Token

A token is an internal object the transformer processes.

A legal move usually has both:

```text
a LEGAL token inside the model
a legal row in legal_qr outside the model
```

They correspond, but they are not the same thing. The token is a learned vector.
The row is the table entry that tells you what an output column means.

### `A`

`A` is the width of the legal-action output table in a batch.

For a single unpadded position:

```text
A = number of legal moves
```

For a padded batch:

```text
A = max legal moves among positions in the batch
legal_mask tells which columns are real for each sample
```

Example:

```text
position 0 has 3 legal moves
position 1 has 5 legal moves

batched policy_place shape = (2, 5)
batched legal_mask shape   = (2, 5)

position 0 mask = [true, true, true, false, false]
position 1 mask = [true, true, true, true, true]
```

## What "Graph" Means Here

In this project, "graph" does not mean the model returns a graph. It means the
input position is represented as a set of named objects and relationships
instead of a fixed image-like board tensor.

A CNN board representation says:

```text
Here are 13 feature planes over a 33x33 grid.
Learn spatial filters over the grid.
```

A global graph representation says:

```text
Here are the actual objects relevant to this state:
- a state token
- player/turn tokens
- stone tokens
- legal move tokens
- tactical window tokens
- line tokens
- cover-set tokens
- pair rows or pair-action tokens

Here are relationships between those objects:
- this legal move is in this window
- this stone is in this window
- this legal move belongs to this cover set
- these tokens are on the same line
- these two rows form a first/second pair relation
```

The transformer trunk then lets these objects exchange information.

So a global graph model is still just a neural net, but its input is closer to:

```text
objects + coordinates + features + relation labels
```

than:

```text
feature planes on a fixed rectangle
```

## What Is Actually Input Into The Graph Model

For the CNN, the main input is easy:

```text
tensors: (B, 13, 33, 33)
```

For the global graph, the main inputs are a bundle of tables:

```text
token_features:      (B, T, 48)
token_type:          (B, T)
token_qr:            (B, T, 2)
token_mask:          (B, T)
legal_token_indices: (B, A)
legal_mask:          (B, A)
relation_type:       (B, T, T)
relation_bias:       (B, 1 or heads, T, T)
```

`T` is the number of graph tokens after padding. `A` is the legal-action table
width after padding.

The model first turns each token row into a vector:

```text
token vector =
  linear(token_features)
  + token_type_embedding(token_type)
  + coordinate_embedding(token_qr)
```

Then relation-biased attention updates the token vectors.

Finally, the model gathers the token vectors for legal moves:

```text
legal_vec = token_vectors[legal_token_indices]
policy_place = linear(legal_vec)
```

This is the closest graph equivalent to the CNN policy head. In the CNN, the
policy head reads a spatial feature at a board cell. In the graph model, the
policy head reads the final token vector for a legal move.

## What Tokens Actually Represent

A token is one object in the position description.

Imagine a position with:

```text
current player: player 0
stones:         (0,0), (1,-1), (2,-2)
legal moves:    (-1,0), (0,1), (3,-3)
tactical facts: one open window, one cover set
```

The graph builder may create tokens like:

```text
token 0: STATE token at (0,0)
token 1: TURN token
token 2: PLAYER token for current player
token 3: PLAYER token for opponent
token 4: STONE token for stone at (0,0)
token 5: STONE token for stone at (1,-1)
token 6: STONE token for stone at (2,-2)
token 7: LEGAL token for legal move (-1,0)
token 8: LEGAL token for legal move (0,1)
token 9: LEGAL token for legal move (3,-3)
token 10: WINDOW6 token for a tactically relevant six-cell window
token 11: COVER_SET token for cells that cover a threat
```

The legal row table then points to the legal tokens:

```text
legal_qr = [
  (-1,0),
  (0,1),
  (3,-3),
]

legal_token_indices = [
  7,
  8,
  9,
]
```

When the model outputs:

```text
policy_place = [0.3, 1.4, -0.2]
```

that means:

```text
row 0 -> token 7 -> move (-1,0) gets logit 0.3
row 1 -> token 8 -> move (0,1) gets logit 1.4
row 2 -> token 9 -> move (3,-3) gets logit -0.2
```

The window and cover tokens are not policy outputs by default. They are context
tokens. Legal move tokens can attend to them, and they can attend back to legal
moves, so tactical structure can influence the final legal move vectors.

## A Token Is An Index Card

The most useful mental model is:

```text
a token = one index card about one object in the position
```

Each card has:

```text
what kind of object it is
where it lives on the board, if it has a location
small numeric facts about that object
```

Then the graph also adds strings between cards:

```text
this legal move is inside this window
this stone is inside this window
this line contains this window
this cover set contains this legal move
these two cells form a pair candidate
these two tokens are nearby or on the same axis
```

The transformer reads the pile of cards and the strings between them. It does
not see a 33x33 image. It sees a list of named objects.

In a CNN, the model might have to learn:

```text
these six cells form a tactical window
```

from local filters over planes.

In the graph model, the graph builder can create:

```text
one WINDOW6 token representing that six-cell window
relations from that WINDOW6 token to the stones and legal moves inside it
```

The model still has to learn whether that window matters, but it does not have
to invent the object from pixels.

## Why These Token Types Exist

The token types are not arbitrary transformer decoration. They are the kinds of
objects the current design thinks may be useful for Hexo search.

### `STATE`

The state token is the global summary slot.

It is similar to a `[CLS]` token in transformer language models. Other tokens
can exchange information with it through attention, and the final state-token
vector feeds heads such as:

```text
value
lookahead_*
regret_value
moves_left
tactical
axis
```

So if `policy_place` is the action output, `STATE` is the position-summary
output.

### `TURN` And `PLAYER`

These tell the model whose perspective the position is from.

The CNN carries this kind of information in feature planes. The graph model
turns it into explicit tokens plus scalar features such as current player and
placements remaining.

### `STONE`

A stone token is one occupied cell.

Conceptually, a stone token says:

```text
there is a stone at (q, r)
it belongs to current player or opponent
it has an age/order in the move history
it has distances to nearby own/opponent stones
it may belong to windows, lines, or components
```

The code stores some of this directly as token features:

```text
q, r, q+r
owner relative to current player
stone age
nearest own/opponent/any stone distances
global own/opponent stone counts
```

Then relations add structure around the stone:

```text
STONE_IN_WINDOW6
SAME_LINE
SAME_AXIS
SAME_COMPONENT
AGE_ORDER_BUCKET
RECENT_MOVE_RELATION
DISTANCE_BUCKET
DIRECTION_BUCKET
```

So a stone token is not just "cell occupied." It is a movable piece of evidence
that can talk to legal moves, tactical windows, components, and the state
summary.

Why have stone tokens instead of only aggregate features?

Because individual stones matter. A single recent stone may create a threat, a
block, a component connection, or a second-placement context. With stone tokens,
attention can ask:

```text
which stones are relevant to this legal move?
which recent stones changed the tactical situation?
which components or windows include this stone?
```

### `LEGAL`

A legal token is one possible action.

This is the global graph equivalent of a CNN cell feature before the policy
head. The important difference is that only legal actions get policy rows.

A legal token says:

```text
this move at (q, r) is legal right now
here are local/tactical scalar facts about it
here are relations to stones/windows/cover sets/lines
```

The final vector for each `LEGAL` token is what `policy_place` scores.

### `WINDOW6`

A `WINDOW6` token represents a six-cell line segment.

In Hexo, six-in-a-row structure is fundamental. A window token can carry facts
such as:

```text
which axis this window is on
how many current-player stones are in it
how many opponent stones are in it
how many empty cells it has
how many legal cells it contains
```

The graph also remembers which cells belong to that window. That lets it create
relations like:

```text
STONE_IN_WINDOW6
LEGAL_IN_WINDOW6
SAME_WINDOW6
```

Conceptually, instead of requiring the network to rediscover every six-cell
segment from spatial convolution, the graph says:

```text
Here is a tactical six-cell segment. These stones and legal moves belong to it.
Decide if it matters.
```

## How `WINDOW6` Is Calculated

A `WINDOW6` token is one contiguous length-6 segment along one of the three hex
line axes.

The builder uses the three forward directions:

```text
(1, 0)
(0, 1)
(1, -1)
```

For an interesting cell `(q, r)`, a length-6 window can contain that cell in
six possible offsets:

```text
cell is position 0 in the window
cell is position 1 in the window
...
cell is position 5 in the window
```

So for each interesting cell, the builder checks:

```text
3 axes * 6 offsets = 18 possible windows containing that cell
```

The interesting cells are:

```text
all occupied stone cells
all current legal move cells
```

The builder does not keep every theoretically possible length-6 segment on the
infinite board. It keeps active windows discovered around stones/legal cells.

For each candidate six-cell segment, it counts:

```text
how many player-0 stones are in the segment
how many player-1 stones are in the segment
which cells in the segment are empty
```

Then it filters:

```text
empty window: discard
window containing both players' stones: discard
window containing stones from only one player: keep
```

That means a `WINDOW6` token is usually an unblocked potential line for one
side. If both players already occupy the same length-6 segment, that segment
cannot become a six-in-a-row for either side without removing stones, so it is
not treated as an active tactical window.

The same physical window can be rediscovered from several stones/legal cells.
The builder deduplicates by:

```text
(axis, window_start_coordinate)
```

So the final token set contains one token per selected active window, not one
token per discovery.

## Why There Are Not Thousands Of Window Tokens

On a midgame board, many stones and legal moves can imply many possible
windows. A single stone can belong to many length-6 segments:

```text
3 axes * 6 offsets = up to 18 windows for one stone
```

But several things reduce the final count:

- duplicate windows are merged by `(axis, start)`;
- empty windows are discarded;
- blocked windows containing both players are discarded;
- if `max_context_tokens` is set, windows are ranked and capped.

When a context budget exists, the window cap is:

```text
window_limit = max(16, context_budget // 4)
```

If there are more active windows than that, the builder ranks them by:

```text
1. windows touching important tactical cells first
   - win-now cells
   - forced-block cells
   - open-four/open-five cells
   - cover cells
2. windows with more stones for either player
3. windows with more total stones
4. windows closer to board center
5. deterministic start-coordinate tie-breakers
```

So in an unconstrained debug/contract setting, the graph can preserve many
active windows. In a budgeted runtime setting, it keeps the tactically dense and
important windows first.

## How One Stone Can Be In Many Windows

One stone token can relate to many `WINDOW6` tokens.

Example:

```text
STONE token S = stone at (0, 0)

WINDOW6 token W1 = axis 0, start (-2, 0)
WINDOW6 token W2 = axis 0, start (-1, 0)
WINDOW6 token W3 = axis 1, start (0, -3)
WINDOW6 token W4 = axis 2, start (-2, 2)
```

If all those windows include `(0, 0)`, the graph records membership:

```text
S belongs to W1
S belongs to W2
S belongs to W3
S belongs to W4
```

In the relation matrix, that becomes multiple relations:

```text
W1 <-> S: STONE_IN_WINDOW6
W2 <-> S: STONE_IN_WINDOW6
W3 <-> S: STONE_IN_WINDOW6
W4 <-> S: STONE_IN_WINDOW6
```

This is not a problem. Attention is naturally many-to-many. The stone token can
send information to many windows, and a legal move token can read from many
windows.

This is one of the reasons a graph is attractive: the same object can
participate in many tactical structures without duplicating the object itself.

## How A Window Relates To Legal Moves And Stones

Each `WINDOW6` token stores the set of six cells it covers.

After all tokens are created, the relation builder asks:

```text
which STONE tokens have coordinates inside this window's six cells?
which LEGAL or HOT_CELL tokens have coordinates inside this window's six cells?
```

Then it creates relations:

```text
WINDOW6 <-> STONE: STONE_IN_WINDOW6
WINDOW6 <-> LEGAL: LEGAL_IN_WINDOW6
tokens in same window: SAME_WINDOW6
```

If a line token summarizes the same axis/line as the window, it also creates:

```text
LINE <-> WINDOW6: LINE_TO_WINDOW6
```

Conceptually:

```text
The window token is a tactical group object.
The stone/legal tokens are individual cell objects.
Relations tell the model which individual cells participate in which groups.
```

The model can then learn patterns like:

```text
legal move L is in a window with 4 friendly stones and 1 empty cell
legal move L is also in a cover set
the recent opponent stone is in an intersecting window
therefore L deserves a high policy score
```

### `LINE`

A `LINE` token summarizes a whole board line along one of the three hex axes.

It can carry facts like:

```text
axis
own stones on the line
opponent stones on the line
longest own run
longest opponent run
```

Line tokens are coarser than `WINDOW6` tokens. A window is a local tactical
segment; a line is a broader directional summary.

Relations connect lines to windows:

```text
LINE_TO_WINDOW6
SAME_AXIS
SAME_LINE
```

This gives the model both local and broader directional context.

### `COVER_SET`

A cover-set token represents a group of legal cells that cover or block a
tactical threat.

It can carry facts like:

```text
how many cells are in the cover set
how many are forced/cover-relevant cells
```

Relations connect it to legal moves and windows:

```text
LEGAL_IN_COVER_SET
WINDOW6_TO_COVER_SET
PAIR_COVERS_THREAT_SET
```

Conceptually, a cover-set token says:

```text
these legal moves are related because together they answer this tactical
problem
```

### `COMPONENT`

A component token summarizes a connected component of stones.

It carries component size and membership relations. This helps the model reason
about connected structures without having to infer every component from scratch.

### `PAIR_ACTION`

A pair-action token can represent a pair as a context object.

The system often avoids materializing all pair-action tokens because pair rows
can be O(A^2), which is expensive. Pair heads can score pair rows by referencing
existing legal/stone tokens instead.

When pair-action tokens are materialized, they can connect to:

```text
their first legal/stone token
their second legal token
cover sets they satisfy
```

This is the most explicit pair representation, but not always the cheapest.

## How A Legal Move Uses Window And Line Tokens

Take one legal move:

```text
LEGAL token L = move at (0, 1)
```

Suppose `(0, 1)` belongs to two relevant six-cell windows and one cover set:

```text
WINDOW6 token W1
WINDOW6 token W2
COVER_SET token C1
LINE token R1
```

The relation table can mark:

```text
L <-> W1: LEGAL_IN_WINDOW6
L <-> W2: LEGAL_IN_WINDOW6
L <-> C1: LEGAL_IN_COVER_SET
R1 <-> W1: LINE_TO_WINDOW6
R1 <-> W2: LINE_TO_WINDOW6
```

During attention, the legal token's query can attend to those context tokens.
The relation labels and relation bias make those connections easier to use.

After several graph blocks, the legal token vector has mixed in information
from:

```text
nearby stones
tactical windows containing it
cover sets containing it
line summaries related to those windows
the global state token
```

Then `policy_place` scores that final legal token vector.

So the output logit for `(0, 1)` is not based only on the coordinates `(0, 1)`.
It is based on the final representation of that legal action after it has read
from the other relevant object tokens.

## How Relations Are Defined

Relations are generated by the graph builder before the model runs.

For every pair of tokens `(i, j)`, the graph builder can assign:

```text
relation_type[i, j]
relation_bias[i, j]
```

`relation_type` is a categorical label like:

```text
same line
same axis
legal move belongs to window
legal move belongs to cover set
stone belongs to window
recent move relation
first/second pair relation
```

`relation_bias` is a numeric attention bias. The base bias is distance-shaped:
nearby coordinates get a stronger default connection than distant coordinates.

Inside attention, this modifies attention scores before softmax:

```text
attention_score(i, j)
  = query(i) dot key(j)
  + learned_embedding_for_relation_type(i, j)
  + relation_bias(i, j)
```

That means the model is not expected to rediscover every board relationship
from coordinates alone. It is told useful facts such as:

```text
this LEGAL token is in this WINDOW6 token
this COVER_SET token contains this legal move
these two tokens are on the same line
this pair row connects these two action tokens
```

The model still learns how much those facts matter.

## What "Global" Means Here

"Global" means the policy is keyed to the full legal action table, not to a
local crop or bounded candidate list.

The crop CNN can score a fixed board grid, and sparse heads can score selected
candidates. The global graph path tries to preserve every legal move as an
explicit legal row:

```text
legal row 0 -> legal move A
legal row 1 -> legal move B
legal row 2 -> legal move C
...
```

MCTS consumes those same legal rows. That is the point. The policy output is
already aligned to the legal actions search is allowed to expand.

The graph may still have capacity limits and padding for batching/IPC, but the
semantic goal is not "score a local crop" or "score a sampled candidate set."
The goal is:

```text
score the current state's legal action rows directly.
```

## What A Row Is

A row is one entry in a table that gives a tensor column semantic meaning.

For dense CNN policy, the row table is implicit and fixed:

```text
row/column 544 -> board index 544
```

For global graph policy, the row table is explicit and per-state:

```text
legal row 0 -> q=0,  r=0
legal row 1 -> q=1,  r=-1
legal row 2 -> q=3,  r=2
```

For opponent policy, it is a different row table:

```text
opponent legal row 0 -> q=-1, r=2
opponent legal row 1 -> q=0,  r=3
```

For pair policy, a row can contain two actions:

```text
pair row 0 -> first=(0, 0), second=(1, -1)
pair row 1 -> first=(0, 0), second=(3, 2)
pair row 2 -> first=(1, -1), second=(3, 2)
```

The neural network output is just logits:

```text
[2.1, -0.3, 0.8]
```

The row table tells you what those logits mean.

This is why row identity matters so much. If the logits and row table get out
of sync, the model can confidently score the wrong moves.

## Does The Model Output A Policy "As Tokens"?

No. More precisely:

```text
The model uses tokens internally.
The model outputs tensors.
Some tensor columns correspond to token-backed rows.
```

For `policy_place`:

1. The graph batch creates one `LEGAL` token for each legal move.
2. The transformer updates all token vectors.
3. The model gathers the final vectors for the legal tokens.
4. A linear head maps each legal token vector to one logit.
5. The result is `policy_place`, shaped like `(B, A)`.

So the policy is not a list of token objects. It is a list of scores over legal
rows, and those legal rows are backed by `LEGAL` tokens.

In CNN terms:

```text
CNN:
  board feature at fixed cell -> policy conv/linear -> logit for fixed cell

Global graph:
  LEGAL token vector for legal row -> linear head -> logit for that legal row
```

That is the clean translation.

## Why A Global Graph Exists

The CNN representation is simple and fast. It is good at local spatial pattern
recognition because every layer shares filters across the crop. For Hexo, that
is a very strong baseline: lines, local threats, nearby stones, and tactical
motifs are naturally spatial.

But the CNN has some friction:

- The model sees a crop, while search ultimately needs legal action rows.
- Sparse/candidate policy heads need extra candidate projection machinery.
- Global relationships such as "these two legal moves together cover a threat"
  are not first-class objects.
- Pair moves are awkward because the natural object is not one cell but a row
  like `(first_q, first_r, second_q, second_r)`.
- Global graph variants may want to carry legal rows, opponent legal rows,
  pair rows, tactical windows, cover sets, components, and relation labels as
  explicit objects.

The global graph path makes those objects explicit.

## Core Concept: Tokens Versus Rows

There are two related but different things:

```text
tokens
rows
```

Tokens are what the transformer-like trunk attends over.

Rows are what heads score and what losses/search consume.

Some rows are backed by tokens, especially legal action rows. Some rows are
represented by references to tokens, especially pair rows.

The most common mistake is to merge these ideas. A `LEGAL` token is an internal
object the transformer can attend to. A legal row is the external table entry
that says a policy column means `(q, r)`. They usually line up, but they are
not the same abstraction.

## What A Graph Batch Contains

The graph batch is defined in `Python/src/hexorl/graph/batch.py`.

A single position carries:

```text
token_features
token_type
token_qr
token_mask
legal_token_indices
legal_qr
legal_mask
opp_legal_qr
opp_legal_mask
pair_first_indices
pair_second_indices
pair_token_indices
relation_type
relation_bias
policy_target
opp_policy_target
pair_first_policy_target
pair_policy_target
pair_second_policy_target
tactical_target
placements_remaining
current_player
```

The important part is that the batch preserves semantic tables instead of
collapsing everything into one board-shaped tensor.

## Token Types

Graph tokens are typed. The current token types are:

```text
STATE
TURN
PLAYER
STONE
LEGAL
HOT_CELL
WINDOW6
LINE
COVER_SET
COMPONENT
PAIR_ACTION
```

Conceptually:

- `STATE` is the pooled state token. Its final vector feeds value-like heads.
- `TURN` and `PLAYER` tell the model whose turn/state it is.
- `STONE` tokens represent occupied cells and ownership/history.
- `LEGAL` tokens represent legal actions. These are the primary policy rows.
- `HOT_CELL` tokens are tactically interesting cells.
- `WINDOW6` tokens represent six-in-a-row windows, the local tactical structure
  Hexo cares about.
- `LINE` tokens summarize whole board lines.
- `COVER_SET` tokens represent sets of cells that cover or block a tactical
  threat.
- `COMPONENT` tokens summarize connected stone components.
- `PAIR_ACTION` tokens can materialize a pair row as a token, though pair heads
  can also score pair references without adding pair tokens to the context.

This is one of the major differences from a CNN. In the CNN, a threat window is
implicit in activations. In the graph model, a threat window can be a named
token with relations to legal moves and stones.

## Token Features

Every token has a fixed-width feature vector. The model also adds:

- a learned token-type embedding;
- a coordinate embedding from `(q, r, q + r)`;
- relation-biased attention between tokens.

This gives the model three sources of identity:

```text
what kind of thing the token is
where it is on the hex board
what scalar features the graph builder attached to it
```

Examples of scalar information include coordinate normalization, distance,
ownership, turn/placement state, tactical counts, window occupancy, cover-set
size, and pair-specific features.

## Relation Types

The graph model does not rely only on raw coordinates. It can receive a
relation matrix between tokens.

Current relation labels include:

```text
DISTANCE_BUCKET
DIRECTION_BUCKET
SAME_AXIS
SAME_LINE
SAME_WINDOW6
STONE_IN_WINDOW6
LEGAL_IN_WINDOW6
LEGAL_IN_COVER_SET
WINDOW6_TO_COVER_SET
LINE_TO_WINDOW6
LEGAL_TO_PAIR_ACTION
PAIR_COVERS_THREAT_SET
SAME_COMPONENT
AGE_ORDER_BUCKET
RECENT_MOVE_RELATION
FIRST_SECOND_PAIR_RELATION
D6_ORBIT_RELATION
```

The attention block can use both:

- `relation_type`, which is embedded per attention head;
- `relation_bias`, which is an additive attention bias.

So a legal move and a window token are not just two arbitrary tokens with
nearby coordinates. The model can be told "this legal move is in this
WINDOW6", or "this pair covers this cover set", or "these tokens are in the
same component."

That is the graph-model bet: give the network explicit semantic handles for
tactical structure while still letting attention learn which handles matter.

## Forward Pass, Conceptually

The shared global graph trunk does this:

1. Project token features to `channels`.
2. Add learned token-type embedding.
3. Add coordinate embedding.
4. Mask padded tokens.
5. Run several relation-biased transformer blocks.
6. Normalize final token vectors.
7. Read the `STATE` token as the state vector.
8. Gather legal token vectors through `legal_token_indices`.
9. Apply variant-specific legal-action processing.
10. Produce only the requested output heads.

The most important readouts are:

```text
policy_place: logits over legal rows
value: binned value from the STATE token
```

For global graph self-play, these replace the dense CNN policy/value path.

## How This Differs From CNN Heads

In the CNN:

```text
policy logits shape: (B, 1089)
meaning: fixed crop board index
```

In the global graph:

```text
policy_place logits shape: (B, A)
meaning: A legal rows for that batch item
```

`A` is not a universal constant. It depends on the state, and batching pads it
with `legal_mask`.

This makes masks and row identity non-negotiable. A graph policy logit without
its legal row table is incomplete.

## Shared Global Graph Heads

All global graph recipes use the same broad head vocabulary.

### `policy_place`

Primary search policy.

It scores `LEGAL` rows:

```text
legal_qr[i] = (q, r)
policy_place[i] = logit for placing at that legal cell
```

Invalid padded rows are masked to a large negative value.

### `value`

Primary MCTS value.

It reads the final `STATE` token and returns a 65-bin value distribution. This
is the value head that must not be disabled for self-play.

### `opp_policy`

Opponent-policy auxiliary head.

It scores an independent opponent legal row table, not the source legal table.
This matters because "what the opponent can do next" may be keyed to a
different legal table after a passive placeholder or future state transition.

### `lookahead_*`

Dynamic value-like heads such as:

```text
lookahead_4
lookahead_12
lookahead_36
```

They read the state vector and predict future value targets at configured
horizons.

### `regret_rank` and `regret_value`

Auxiliary regret heads from the state vector.

They are training signals, not search policy outputs.

### `moves_left`

Predicts remaining game length or move count scale from the state vector.

### `tactical`

State-level tactical classifier. Current target shape is four labels:

```text
win-now
must-block
cover-pair
quiet
```

### `axis` and `axis_delta_norm`

Auxiliary spatial/axis outputs. `axis_delta_norm` emits a map-like tensor, even
though it comes from the graph state vector.

### `legal_token_quality`

Auxiliary legal-row scorer. It scores legal token rows, like `policy_place`,
but is a diagnostic/quality-style signal rather than the primary MCTS policy.

## Registered Global Graph Recipes

All recipes below are registered in `Python/src/hexorl/models/registry.py` and
built through `GlobalHexGraphNet`.

They share:

- `input_contract_id = global_graph_tokens:v1`;
- default outputs `policy_place` and `value`;
- global legal-row policy provider;
- graph pair capability metadata;
- replay sparse diagnostics;
- attention-head divisibility requirement.

They differ by relation requirements, depth behavior, and variant-specific
legal-action processing.

## `global_graph_option1`: Relation Graph Baseline

Registry metadata:

```text
architecture_id: global_graph_option1
family_id: relation_graph
relation_required: true
```

Conceptual role:

```text
The baseline relation-aware global graph.
```

It uses the shared token trunk and requires relation tensors. There is no extra
variant-specific legal-action branch after the trunk.

Use this mental model:

```text
Let all semantic tokens attend to each other with typed relation information,
then score legal moves from the resulting LEGAL token vectors.
```

This is the cleanest "pure relation graph" baseline. It should tell you whether
explicit token and relation structure is useful before adding more specialized
heads or gates.

## `global_xattn_0`: Legal-To-Context Cross Attention

Registry metadata:

```text
architecture_id: global_xattn_0
family_id: context_cross_attention
relation_required: false
```

Conceptual role:

```text
A lighter global graph where legal actions explicitly query context tokens.
```

Implementation differences:

- Caps graph block count to at most 2.
- After the shared trunk, legal action vectors run an extra cross-attention
  pass over non-legal context tokens.
- The context excludes `LEGAL` tokens, so each legal row asks: "given the
  state, stones, windows, lines, covers, and other context, what should I know?"

Compared with `global_graph_option1`:

- less deep relation processing;
- more explicit legal-action readout;
- potentially cheaper and more focused.

Use this mental model:

```text
First build context tokens, then let each legal action query that context.
```

This is close to the transformer idea of decoder queries attending to encoder
memory, except the queries are legal moves.

## `global_line_window_0`: Tactical Line/Window Gating

Registry metadata:

```text
architecture_id: global_line_window_0
family_id: line_window_cover
relation_required: true
```

Conceptual role:

```text
A relation graph that gives legal actions an explicit tactical summary.
```

Implementation differences:

- Requires relation tensors.
- After the shared trunk, it selects tactical token types:
  - `WINDOW6`
  - `LINE`
  - `COVER_SET`
- It averages their final token vectors into one tactical context vector.
- Each legal vector receives a learned gate from:

```text
[legal_vec, tactical_context]
```

This means every legal action can be adjusted by a board-level tactical summary
of lines, windows, and threat-cover structures.

Use this mental model:

```text
Score legal moves after injecting an explicit summary of tactical line/window
pressure.
```

This recipe is for testing whether tactical graph tokens should have a direct
path into action scoring rather than relying entirely on generic attention.

## `global_pair_twostage_0`: Pair-Specific Two-Stage Variant

Registry metadata:

```text
architecture_id: global_pair_twostage_0
family_id: pair_two_stage
relation_required: false
```

Conceptual role:

```text
A global graph specialized for two-placement pair reasoning.
```

Hexo has phases where a move can involve first and second placement logic. A
single-cell policy can miss interactions where a pair is strong even if either
cell alone is not obviously best.

This recipe adds pair-specific refinement modules:

```text
pair_first_refine
pair_second_refine
```

They are used only by `global_pair_twostage_0`.

The first refinement conditions legal action vectors on the state:

```text
first_action_representation + f(first_action_representation, state)
```

The second refinement conditions the second action on:

```text
second_action_representation
first_action_representation
state
```

That gives the model a structured way to distinguish:

```text
Which first placement looks promising?
Given the known first placement, which second placement completes it?
Which unordered pair is jointly promising?
```

Use this mental model:

```text
First reason about candidate first placements, then reason about second
placements conditioned on the first.
```

This is the most explicitly pair-aware global recipe.

## `global_graph_full_0`: Full Relation Graph Slot

Registry metadata:

```text
architecture_id: global_graph_full_0
family_id: full_relation_graph
relation_required: true
```

Conceptual role:

```text
A named full-relation variant.
```

In the current implementation, this uses the same forward-path branch as
`global_graph_option1`: shared relation-biased graph blocks and standard legal
readout.

Its practical distinction today is metadata and required relation tensors. It
is a reserved slot for a heavier or more complete relation schema, not yet a
deeply distinct architecture in Python code.

Use this mental model:

```text
Same current mechanics as the relation baseline, intended as the full-capacity
relation-family name.
```

## `global_hybrid_action_0`: Graph With Action Feature/Crop Context

Registry metadata:

```text
architecture_id: global_hybrid_action_0
family_id: crop_diagnostic_global_action
relation_required: false
```

Conceptual role:

```text
A bridge between pure global graph action rows and crop/action feature context.
```

Implementation differences:

- After the trunk, it gathers raw feature vectors for each legal token.
- Each legal vector receives a learned gate from:

```text
[legal_vec, raw_legal_features]
```

- If a `crop_tensor` is provided, it averages the crop planes and injects a
  crop-context vector into legal rows.

This lets the global graph model use legal-row identity while still accepting
extra crop-derived context when available.

Use this mental model:

```text
Global legal-row scoring with optional crop-informed action context.
```

This is a useful experiment if you suspect the pure token graph loses some
signal that the CNN crop representation captures cheaply.

## `global_graph768_champion`: Scaled Relation Graph

Registry metadata:

```text
architecture_id: global_graph768_champion
family_id: scaled_relation_graph
relation_required: true
```

Conceptual role:

```text
A scaled-up relation graph candidate.
```

Implementation differences:

- Requires relation tensors.
- Forces at least 6 graph blocks.
- Intended to pair with larger graph token budgets such as 768.

Use this mental model:

```text
The larger relation-graph recipe, meant for more capacity and richer token
sets.
```

It is not a completely different mechanism; it is the scaled version of the
relation graph family.

## `graph_hybrid_0`: Not A Global Graph

This one is easy to confuse with the global graph recipes.

Registry metadata:

```text
architecture_id: graph_hybrid_0
family_id: crop_sparse_graph_hybrid
global_graph: false
```

Conceptual role:

```text
A crop-compatible sparse graph scout.
```

It still builds through the dense `HexNet` family and still operates on the
crop tensor. It can use sparse candidate policy and crop pair policy, but it is
not a legal-row global graph model.

Use this mental model:

```text
CNN/crop model with graph-flavored sparse candidate machinery.
```

It is not in the same category as `global_xattn_0`,
`global_line_window_0`, or `global_pair_twostage_0`.

## Pair Heads

Pair heads are separate from pair strategy.

The heads produce logits. The strategy decides whether and how those logits
influence search.

This distinction is important:

```text
Head exists != MCTS uses it
```

MCTS pair influence requires an explicit `model.pair_strategy`: `none`,
`root_pair_mcts`, or `full_pair_mcts`.

## Why Pair Heads Exist

A normal policy head scores one legal cell at a time.

Pair reasoning asks a different question:

```text
How good is this pair of placements together?
```

That matters when:

- the first placement creates or blocks a structure only with a second
  placement;
- the second placement is conditioned on a known first placement;
- two mediocre-looking individual cells are strong together;
- tactical cover pairs need to be represented directly.

## Pair-Phase Vocabulary

The graph code distinguishes two broad pair phases.

### First-placement or unordered pair phase

There are at least two placements remaining.

The pair row is an unordered pair of legal cells:

```text
(first_q, first_r, second_q, second_r)
```

For target purposes, `(a, b)` and `(b, a)` represent the same unordered pair.

### Second-placement or known-first phase

There is one placement remaining in the current turn, and the first placement
is already known.

The pair row means:

```text
(known_first_q, known_first_r, candidate_second_q, candidate_second_r)
```

Now order matters. The first coordinate is fixed by the already-selected first
placement; the model scores legal second placements.

## `policy_pair_first`

Shape:

```text
(B, A)
```

where `A` is the number of legal rows.

Meaning:

```text
How promising is each legal action as the first placement of a pair?
```

This head is masked to active first-pair phases. It is a marginal projection of
pair target mass onto legal first-placement rows.

Conceptually:

```text
Before choosing the whole pair, which first move points toward strong pairs?
```

In `global_pair_twostage_0`, this head uses the pair-first refinement path.

## `policy_pair_joint`

Shape:

```text
(B, P)
```

where `P` is the number of pair rows.

Meaning:

```text
How promising is this unordered pair of placements?
```

For first-placement pair phases, it uses pair rows built from legal token
references. Its feature construction combines:

```text
state
first + second
abs(first - second)
first * second
```

That gives the scorer symmetric pair information. The pair should score the
combination, not just concatenate two arbitrary ordered actions.

Conceptually:

```text
Score the pair as a joint object.
```

## `policy_pair_second`

Shape:

```text
(B, P)
```

Meaning:

```text
Given a known first placement, how good is each legal second placement?
```

It is used for known-first pair rows, where the first coordinate is already
part of the current state/turn context.

Its feature construction combines:

```text
state
first_conditioned_vector
second_conditioned_vector
abs(first - second)
```

In `global_pair_twostage_0`, `pair_second_refine` makes this explicitly
conditional on first placement and state.

Conceptually:

```text
The first move is known. Complete the pair.
```

## Crop `pair_policy`

The older crop-compatible pair head is different from graph pair heads.

In dense/crop models, `pair_policy` scores selected candidate pair rows. It is
not a global legal-row pair table. It depends on candidate rows supplied to the
crop model.

Conceptually:

```text
Score pairs inside a bounded candidate set.
```

Graph pair heads instead score graph row tables keyed by legal and pair rows.

## Pair Strategy

Current pair strategy options:

```text
none
root_pair_mcts
full_pair_mcts
```

`none` means pair heads may exist for training or diagnostics, but search does
not use them.

`root_pair_mcts` means pair priors are projected only at the root before MCTS
expands the first search frontier.

`full_pair_mcts` means pair priors apply at the root and at supported non-root
search points. Unsupported model/search combinations are rejected before Rust
MCTS calls.

Both pair-enabled modes require:

```text
pair_strategy_max_pairs > 0
pair_prior_mix > 0
```

It declares required output contracts:

```text
pair_policy
policy_pair_first
policy_pair_joint
policy_pair_second
```

In practice, graph pair scoring happens in bounded chunks. The strategy:

1. Builds or patches graph batches with pair rows.
2. Sends those rows through inference.
3. Reads `policy_pair_joint` or `policy_pair_second`.
4. Projects pair logits back into action priors when needed.
5. Blends those priors with base action logits using `pair_prior_mix`.

This is deliberately outside the model class. The model produces pair logits;
the strategy owns runtime use.

## How Base Policy Reaches MCTS

For normal global graph search, the handoff is:

```text
graph builder creates legal_qr
model outputs policy_place logits aligned to legal_qr
inference adapter returns logits + row metadata
self-play/search validates the rows against Rust legal moves
MCTS uses the logits as priors for child actions
```

If the legal row table is:

```text
0 -> (0, 0)
1 -> (1, -1)
2 -> (3, 2)
```

and `policy_place` is:

```text
[0.2, 1.7, -0.4]
```

then MCTS sees priors for these actions:

```text
(0, 0)  gets logit 0.2
(1, -1) gets logit 1.7
(3, 2)  gets logit -0.4
```

MCTS does not choose the move by simply taking the highest policy logit. The
policy is a prior. Search then runs simulations using:

```text
prior probability
visit counts
Q/value estimates
exploration terms
legal move constraints
temperature/final selection rules
```

At the end, the actual chosen move usually comes from the MCTS visit
distribution, not raw model argmax.

So the global graph policy does this:

```text
suggest promising legal actions to search
```

and MCTS does this:

```text
test those actions through tree search and choose from the resulting visits
```

## How A Pair Logit Becomes An Action Prior

MCTS usually expands single legal actions. Pair logits are over pair rows.

To apply pair knowledge at a single-action root, the strategy can project pair
probability mass back to legal actions:

```text
pair row (a, b) has probability p
add p to action a
add p to action b
normalize action mass
blend with base policy
```

For known-first second-placement, the projection mostly affects possible
second placements.

This is not the same as replacing the policy head. It is an auxiliary prior
mixed into action selection.

## How Pair Heads Affect The Actual Move

The base `policy_place` head is already enough for MCTS to choose legal moves.
Pair heads are extra information.

With `pair_strategy = none`:

```text
pair heads may train
pair heads may be logged
MCTS ignores pair heads
actual move selection uses normal policy/value/search
```

With `pair_strategy = root_pair_mcts` or `pair_strategy = full_pair_mcts`:

```text
pair heads can contribute extra priors
```

There are two main cases.

### First Placement Of A Pair Turn

Suppose the legal moves are:

```text
A = (0, 0)
B = (1, -1)
C = (3, 2)
```

The base policy scores individual moves:

```text
A: 0.2
B: 1.7
C: -0.4
```

The pair strategy may also score pair rows:

```text
(A, B): 2.0
(A, C): 0.1
(B, C): 1.2
```

Those pair scores are projected back onto individual first actions. If `(A, B)`
is strong, both `A` and `B` receive some extra prior mass. If `(B, C)` is also
strong, `B` receives mass from both pairs.

Then the strategy blends:

```text
final action prior = mostly base policy + some pair-derived prior
```

The exact blend is controlled by `pair_prior_mix`.

MCTS still picks one legal action at the root. The pair head does not directly
force a two-action choice. It biases the first action toward moves that
participate in strong pairs.

### Second Placement With Known First

After the first placement is already known, the pair rows become:

```text
(known_first, A)
(known_first, B)
(known_first, C)
```

Now `policy_pair_second` is naturally aligned with possible second moves. The
pair strategy can use those logits to bias the second-placement action prior.

This is simpler than the first-placement case because the first action is
fixed. The pair head is answering:

```text
Given the first placement we already made, which legal second placement best
completes the pair?
```

MCTS still runs search. The pair head changes priors; it does not bypass the
tree.

## How Pair Heads Train Versus How They Are Used

Training and search are separate.

During training:

```text
policy_pair_first learns marginal first-placement targets
policy_pair_joint learns unordered pair targets
policy_pair_second learns known-first second-placement targets
```

During search:

```text
pair_strategy decides whether to request/use pair outputs
pair_strategy chunks pair rows if needed
pair_strategy blends pair-derived priors into MCTS
```

This separation is intentional. It lets you train pair heads for diagnostics or
future experiments without accidentally changing self-play behavior.

## How To Think About The Recipes

Here is the conceptual lineup.

| Recipe | Main idea | Relation tensors | Distinct code path today |
|---|---|---:|---|
| `global_graph_option1` | baseline relation graph | required | shared trunk only |
| `global_xattn_0` | legal actions query context tokens | optional | legal cross-attention |
| `global_line_window_0` | inject tactical line/window/cover context | required | tactical gate |
| `global_pair_twostage_0` | first/second pair-specific refinement | optional | pair refinement MLPs |
| `global_graph_full_0` | full relation graph slot | required | currently shared trunk only |
| `global_hybrid_action_0` | legal rows plus raw action/crop context | optional | hybrid action gate |
| `global_graph768_champion` | scaled relation graph | required | depth floor of 6 blocks |
| `graph_hybrid_0` | crop-compatible sparse graph scout | not global | dense `HexNet` path |

## Practical Intuition

If you are comparing these experimentally:

- Start with `global_graph_option1` if you want the clean relation-graph
  baseline.
- Try `global_xattn_0` if you want a lighter model where legal actions
  explicitly attend to context.
- Try `global_line_window_0` if you think tactical window and cover structure
  should directly steer action scoring.
- Try `global_pair_twostage_0` if pair reasoning is the core experiment.
- Treat `global_graph_full_0` as a full-relation family slot, not yet a deeply
  distinct model.
- Try `global_hybrid_action_0` if you want to preserve some crop/action-feature
  signal while using legal-row graph policy.
- Use `global_graph768_champion` when testing a larger token-budget/deeper
  relation graph.
- Use `graph_hybrid_0` only when you want crop-compatible sparse candidate
  behavior, not true global legal-row behavior.

## The Most Important Correctness Rule

For CNN policy, a logit index has stable meaning because the board grid is
fixed.

For global graph policy, a logit index has meaning only with its row table.

This is why the refactor emphasizes contracts:

```text
logits + row table + mask + phase + value decoder
```

Without those, a graph model can appear to train while silently learning the
wrong target. The whole design is trying to make that failure mode hard.
