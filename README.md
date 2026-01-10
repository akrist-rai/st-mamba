🔍 Key Engineering Details

    reshape vs view:

        In Phase 1, I used view because we just stacked layers. The memory was already clean.

        In Phase 2, I used reshape because permute messed up the memory layout. reshape automatically handles the .contiguous() fix for you.

    Broadcasting adj_matrix:

        Notice self.spatial_layer(x_slices, adj_matrix).

        x_slices has size B*T (e.g., 384 graphs).

        adj_matrix has size 1 (1 map).

        PyTorch automatically applies that 1 map to all 384 graphs. This is super efficient.

    Residual Connections:

        I added x + output at every step. This is crucial. Without it, the model forgets the "Space" info while learning the "Time" info, and vice versa.




1. The Old Way (Sequential)
I process all the space first, then all the time. They don't talk to each other frequently.
Code snippet

graph LR
    Input --> GAT_Layer1 --> GAT_Layer2 --> Mamba_Layer1 --> Mamba_Layer2 --> Output

2. The New Way (Interleaved / Mixed)

What the STMambaBlock does. We package Space and Time into a single atom. We repeat this atom.
