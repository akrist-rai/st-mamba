🧊 Understanding the Data Geometry: The 4D Hypercube

In high-performance Deep Learning, we rarely work with simple 2D tables. Instead, we model the world as a Multi-Dimensional Tensor.

For Spatio-Temporal problems (like Traffic, Weather, or Video), our data takes the shape of a 4-Dimensional Hypercube.
1. Visualizing the Dimensions

You can imagine the input tensor [Batch, Time, Nodes, Features] not as a flat list, but as a stack of dynamic 3D objects.
Dimension	Shape Analogy	What it represents
1D	A Line	A single value over time (e.g., Stock Price).
2D	A Sheet of Paper	A static map or image (e.g., A photo of traffic).
3D	A Cube	A video or history of maps (e.g., 12 seconds of traffic).
4D	A File Cabinet	A Batch of 32 independent traffic simulations running in parallel.
2. Why do we need such complex shapes?

We use this 4D structure because it allows us to isolate and conquer specific relationships using the "Slice & Skewer" technique.

The model is essentially a machine that looks at this Hypercube from two different angles:

    Angle 1 (Space): We slice the cube along the Time Axis.

        The View: We see a static map.

        The Logic: "How does Node A affect Node B right now?" (Handled by GAT).

    Angle 2 (Time): We rotate the cube and look down the Node Axis (Skewering).

        The View: We see a single node's history.

        The Logic: "How does the past affect the future?" (Handled by Mamba).

3. The Engineering Implication

To support this "Multi-View" processing on a GPU, we perform a Dimension Dance:

    Load: Data enters as [Batch, Time, Nodes, Features].

    Flatten: We merge Batch and Time to let the GNN process every snapshot in parallel.

    Pivot (Permute): We physically rotate the tensor in memory (using .permute() and .reshape()) to line up the time-steps for Mamba.

    Scan: Mamba scans the time-axis efficiently in SRAM.

    Restore: We rotate back to the original shape for the final prediction.

    Summary: The 4D Tensor is not just a data container; it is a coordinate system that allows us to mathematically separate "Where" (Space) from "When" (Time).
