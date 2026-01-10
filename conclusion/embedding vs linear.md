🧠 Design Decision: nn.Linear vs nn.Embedding

In the architecture of CommuteGraphMamba, specifically within the GAT (Graph Attention) blocks, we explicitly use nn.Linear layers for our projections (Wq​,Wk​,Wv​) instead of nn.Embedding.

While both layers involve learnable weight matrices, they serve fundamentally different purposes based on the Input Data Type:
1. The Core Difference
Layer	Input Type	Operation	Analogy
nn.Embedding	Discrete Integers (Indices)	Lookup / Selection: Selects a specific row from the matrix.	The Dictionary: You give a page number, it gives you the text on that page.
nn.Linear	Continuous Floats (Vectors)	Mixing / Transformation: Performs a weighted sum of all rows (Matrix Multiplication).	The Blender: You give a recipe of ingredients, it mixes them into a new flavor.
2. Mathematical Equivalence

Mathematically, nn.Embedding is strictly equivalent to nn.Linear (with bias=False) IF the input to the Linear layer is a One-Hot Encoded vector.

    Embedding Lookup: "Get Row 5."

    Linear Multiplication: "Multiply Matrix by [0, 0, 0, 0, 1, 0...]."

        Since the vector is all zeros except for position 5, the math results in extracting Row 5.

3. Why we use nn.Linear in Project Commute

In our Spatio-Temporal block, the input h (hidden state) is a Dense Vector containing rich, continuous information (e.g., specific traffic speeds, rain density, historical patterns).

    State h: [0.55, -1.2, 3.4, ...] (Floats).

    Goal: We need to mix these features to create a new "Query" vector.

    Reasoning: Since h is not a discrete ID (like a Word ID or Node ID) but a continuous feature set, nn.Embedding cannot handle it. We must use nn.Linear to perform the matrix multiplication required to transform the feature space.
