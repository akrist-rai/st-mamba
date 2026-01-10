# Spatio-Temporal Mamba (ST-Mamba) with GAT

This repository implements a **Spatio-Temporal Graph Neural Network** that combines **Graph Attention Networks (GAT)** for spatial dependencies and **Mamba (State Space Models)** for temporal dependencies. 

It is designed for tasks where data has both a graph structure (nodes connected by edges) and a time component (traffic flow, weather forecasting, sensor networks).

## 🧠 Architecture Overview

The core of the model is the `STMambaBlock`, which processes data in an interleaved fashion:
1.  **Spatial Pass (GAT):** Nodes communicate with neighbors at each specific timestep.
2.  **The Pivot:** The data cube is rotated to expose the time dimension.
3.  **Temporal Pass (Mamba):** Each node's history is processed independently using Mamba's linear-time sequence modeling.

### Data Flow Diagram
''' mermaid
graph TD
    Input["Input Tensor\n(B, T, N, F)"] --> View1["View / Flatten\n(B*T, N, F)"]
    View1 --> GAT["Spatial Layer (GAT)\nNodes talk to Neighbors"]
    GAT --> AddNorm1["Add & Norm"]
    AddNorm1 --> View2["View / Restore\n(B, T, N, F)"]
    
    View2 --> Pivot1{"The Pivot\nPermute Dimensions"}
    Pivot1 --> Reshape["Reshape\n(B*N, T, F)"]
    
    Reshape --> Mamba["Temporal Layer (Mamba)\nPast talks to Future"]
    Mamba --> AddNorm2["Add & Norm"]
    
    AddNorm2 --> Restore["Restore Dimensions\n(B, N, T, F)"]
    Restore --> Pivot2{"Pivot Back"}
    Pivot2 --> Output["Output Tensor\n(B, T, N, F)"]
    
    style Pivot1 fill:#f9f,stroke:#333
    style Pivot2 fill:#f9f,stroke:#333
    style GAT fill:#bbf,stroke:#333
    style Mamba fill:#bfb,stroke:#333
'''


