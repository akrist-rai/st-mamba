⚙️ Implementation Details: Inside the GAT Layer

This section details the tensor manipulations used to enable Multi-Head Attention and enforce Graph connectivity constraints.
1. The "Split & Shuffle" (Multi-Head Preparation)

To enable the model to view the graph from multiple perspectives (Heads) simultaneously, we perform a specific sequence of dimension transformations.
```Python

# The Code
q = self.W_q(h).view(B*T, N, self.num_heads, self.head_dim).transpose(1, 2)
```

Step-by-Step Tensor Trace:
Step	Operation	Shape [Dims]	Explanation
1	Input	[Batch, N, Total_Dim]	The dense feature vector for every node.
2	Project	[Batch, N, Total_Dim]	Linear layer transforms features into "Query" space.
3	View (Split)	[Batch, N, Heads, Head_Dim]	The Split. We cut the big vector into 4 smaller, parallel vectors.
4	Transpose (Shuffle)	[Batch, Heads, N, Head_Dim]	The Shuffle. We move Heads to Dim 1 so PyTorch treats them as independent parallel matrices.

    Why? This alignment allows us to compute attention for all 4 heads across all 100 batches in a single matrix multiplication operation.

2. The "Gatekeeper" (Adjacency Masking)

Standard transformers allow every node to talk to every other node (O(N2)). In a Graph Neural Network, nodes must only attend to their physical neighbors. We enforce this using a Mask.
Python

# The Code
mask = (adj == 0).view(1, 1, N, N)
scores = scores.masked_fill(mask, -1e9)

The Logic Flow:

    Boolean Mask (adj == 0): Creates a True/False grid.

        True = Path is Blocked (No road).

        False = Path is Open (Road exists).

    Broadcast (view(1, 1, N, N)): Expands the single city map to apply to every Batch and every Head automatically.

    Mask Fill (-1e9):

        We replace the score of disconnected nodes with Negative One Billion.

        Why? Because Softmax(-1,000,000,000) ≈ 0. This mathematically forces the attention weight to zero, effectively cutting the wire between unconnected nodes.

Visualizing the Mask:
Connection	Adjacency Value	Mask Value	Raw Score	Filled Score	Final Probability (Softmax)
Connected	1	False	2.5	2.5	High (e.g., 0.9)
Disconnected	0	True	1.8	-1e9	Zero (0.0)


