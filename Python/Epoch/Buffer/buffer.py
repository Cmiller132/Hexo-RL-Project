"""BUFFER SPECS (KataGo-Style Continuous Memory)
    Core Philosophy
        Continuous Ring Buffer: Maintain a sliding window (e.g., 1M - 2M samples). Overwrite oldest samples when full.
        RAM-Only: The entire buffer lives in system RAM.
        Training: A "training phase" is simply pulling a fixed-size chunk (e.g., 20k samples) from the buffer. Training time remains constant regardless of total buffer size.
        
    Storage Format (Compact)
        Store ONLY lightweight data: compact move histories, sparse policy targets, scalars, turn boundaries, and PCR flags.
        NO dense tensors in RAM.
        The Rust `pybridge` dynamically re-encodes the (13, 33, 33) board state and applies D6 symmetry on-the-fly during dataloader access.

    Sampling Strategy
        Recency-Biased Uniform Sampling: Select chunks randomly, but mathematically skew the probability toward newer games (e.g., `recency_decay = 0.9`).
        Quality Gating: Differentiate samples generated from full-search vs. shallow-search (PCR) 
        """