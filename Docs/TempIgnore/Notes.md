These are human written do not touch.


Interesting proposals

1. Latent Mixture Of Soft Pointers
2. 


2. Low-Rank Policy Factorization (The Dot-Product Policy)This is the formal mathematical realization of your "Binned Encoder" idea, turning it into a true dense policy target. If MCTS absolutely requires an $N \times N$ probability matrix to query against, you can generate it dynamically using a low-rank approximation.How it works: Your network outputs an embedding tensor of size $N \times D$ (where $N$ is the number of valid hexes and $D$ is a latent dimension, e.g., 64). You can also output a separate scalar marginal logit $M_i$ for each hex.The Output: The network never explicitly constructs the 4D pair tensor. Instead, the raw logit for playing hex $i$ and hex $j$ together is defined as:$$L(i, j) = M_i + M_j + (\mathbf{e}_i \cdot \mathbf{e}_j)$$where $\mathbf{e}_i$ and $\mathbf{e}_j$ are the $D$-dimensional latent vectors for those hexes.Joint Reasoning: The network learns to map synergizing pairs into similar or complementary regions of the $D$-dimensional space.MCTS Integration: When MCTS expands a node, you don't calculate all $N^2$ probabilities. You only calculate $L(i, j)$ for the specific pair action MCTS is currently considering, which requires a trivial dot product. If you need the full matrix for search initialization, $\mathbf{E}\mathbf{E}^T$ yields the $N \times N$ grid in a single highly optimized matrix multiplication.