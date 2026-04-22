"""AlphaZero-style neural network for Hexo

Architecture: pre-activation residual tower with hex-masked convolution,
global pooling layers (mean+max+stddev), and permanent RGSC-compatible heads.
Supports Mish activations, RepVGG-linear hex convolutions, and a FixScale
trunk path inspired by KataGo's fixed-scale residual initialization.

- Input:          (batch, 13, 33, 33) -- board encoding from features.py
- Policy:         (batch, 1089)       -- logits over 33x33 grid
- Value:          (batch, B)          -- B-bin categorical value; V(s) = sum(p_i * v_i)
- AxisInfluence:  (batch, 3, 33, 33)  -- per-axis line influence scores
- OppPolicy:      (batch, 1089)       -- opponent next-policy logits
- RegretRank:     (batch, 1)          -- ranking score for RGSC
- RegretValue:    (batch, 1)          -- absolute regret estimate for RGSC
- MovesLeft:      (batch, 1)          -- predicted moves remaining



VALUE TARGETS (KataGo-Style EMA Lookahead)
1. **Do not use strict ply index targets** as the primary short-term target.

2. **Use stable turn-boundary targets**:
   - Boundary state is the start of a turn (`placements_remaining == 2`).
   - For both placements in a source turn, use the same future boundary sequence.

3. **Use KataGo-style EMA over future search values**, adapted to this game:

   - KataGo formula reference: `docs/KataGoMethods.md` short-term section:
     `(1-lambda) * sum_{t' >= t} MCTS_value(t') * lambda^(t'-t)`.
   - Adaptation here: the sum runs over **future turn boundaries** rather than every ply.

4. **Perspective correctness is mandatory**:
   - Reframe each future boundary value into the source snapshot player's frame before averaging.

5. **Initial horizons**:
   - Two aux heads first: mean horizon about 4 turns and 10 turns.
   - Suggested starting lambdas (turn-boundary domain): `lambda_short ~= 0.75`, `lambda_mid ~= 0.90`.
   - Add a longer third head only after training is stable.

6. **Loss wiring**:
   - Project each EMA scalar target into value bins (same projection as main value head).
   - Train each aux head with CE/KL against binned targets.
   - Start aux loss scales at about `0.15-0.25` each and ramp up over 5-10 epochs.

7. **Quality-weight the EMA terms**:
   - Full-search boundary values receive larger weight than low-sim PCR values.
   - This is required to avoid importing low-sim noise into aux targets.

8. **Edge-case handling**:
   - Opening turn (single placement), PRB restarts, and terminal-after-placement-1 must be handled
     by deriving boundaries from game state fields, not index parity assumptions.
   - Near end of game: renormalize EMA on available terms; if no future term exists, bootstrap from terminal outcome.

#### Why This Is a Good KataGo Adaptation

- Preserves the core KataGo idea: lower-variance temporal targets from future search values.
- Adjusts the time axis to this game's action semantics (two placements per turn), avoiding parity bias
  that does not exist in Go's one-move turns.
- Keeps the same practical role as KataGo's auxiliary TD value heads seen in training output structure:
  `td_value_logits` in `python/katago/train/model_pytorch.py` and TD losses in
  `python/katago/train/metrics_pytorch.py`.



"""