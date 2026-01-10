```python
# 1.3 Apply Graph Mask (Using Adjacency Matrix)

        mask = (adj == 0).view(1, 1, N, N)

        scores = scores.masked_fill(mask, -1e9)
```


"Gatekeeper" of your Graph Neural Network.

It enforces the rule: "You can only talk to your actual neighbors."

Without these two lines, your model is not a Graph Neural Network; it is just a standard Transformer where every node talks to every other node (Global Attention). These lines force the "Graph" structure.

Here is the step-by-step breakdown:
1. The Logic: mask = (adj == 0)

    Input: adj is your Adjacency Matrix (0s and 1s).

        1: Connected (Road exists).

        0: Disconnected (No road).

    Operation: adj == 0 checks for the disconnected spots.

    Result: It creates a True/False grid (Boolean Tensor).

        True: "This path is BLOCKED." (Where adj was 0).

        False: "This path is OPEN." (Where adj was 1).

2. The Shape: .view(1, 1, N, N)

    The Problem:

        Your scores tensor is huge: [Batch, Heads, N, N]. (e.g., 100 batches, 4 heads).

        Your mask tensor is small: [N, N]. (Just one map of the city).

    The Fix: .view(1, 1, N, N) adds two "fake" dimensions to the front.

    Broadcasting: This tells PyTorch: "Take this one city map and apply it to every single Batch and every single Attention Head."

3. The Action: scores.masked_fill(mask, -1e9)

    Goal: We want the attention probability to be 0%.

    The Math: We are about to feed these scores into a Softmax function.

        Softmax formula: ∑exex​

        If we put in 0: e0=1 (100% attention! Bad!).

        If we put in -1,000,000,000: e−1000000000≈0.0000... (0% attention. Perfect!).

    The Command: masked_fill(mask, -1e9) says: "Wherever the mask is True (Blocked), replace the score with negative one billion."

Summary Visualization
Step|	Node A → Node B	Value	|Softmax Result	|Meaning|
Raw Score	|Calculated by Q & K	|2.5	...	|"They seem related."
Adjacency	|Check Map	|0 (No Road)	...	|"But they aren't connected."
Mask	|adj == 0	|True	...	|"Block this!"|
Fill	|Apply -1e9	|-1,000,000,000	...	|"Delete the connection."
Final	|Softmax	...	|0.0	|"Node A ignores Node B."
