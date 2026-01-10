# Spatio-Temporal Mamba (ST-Mamba) with GAT

This repository implements a **Spatio-Temporal Graph Neural Network** that combines **Graph Attention Networks (GAT)** for spatial dependencies and **Mamba (State Space Models)** for temporal dependencies. 

It is designed for tasks where data has both a graph structure (nodes connected by edges) and a time component (traffic flow, weather forecasting, sensor networks).

## 🧠 Architecture Overview

The core of the model is the `STMambaBlock`, which processes data in an interleaved fashion:
1.  **Spatial Pass (GAT):** Nodes communicate with neighbors at each specific timestep.
2.  **The Pivot:** The data cube is rotated to expose the time dimension.
3.  **Temporal Pass (Mamba):** Each node's history is processed independently using Mamba's linear-time sequence modeling.

### Data Flow Diagram

ggraph TD
    %% Define nodes with descriptive text based on the code's logic
    Input(["Start: Raw Input Data\n(Historical Node Features + Graph Structure)"])
    
    %% --- Main Model Level ---
    InitProj["Initial Linear Projection\n(Transform raw features into the model's hidden dimension)"]

    %% --- The ST-Mamba Block Level (Subgraph) ---
    subgraph ST_Block ["ST-Mamba Layer Block (Repeated for N layers)"]
        direction TB
        
        %% Phase 1: Space
        GAT_Input[Input to Block]
        SpatialPass["1. Spatial Pass (GAT Layer)\n'At every specific timestep, nodes communicate with their neighbors using attention.'"]
        AN1(Add & LayerNorm)
        
        %% Phase 2: Pivot
        PivotT{"2. The Pivot (Rotate)\n'Rearrange data to prioritize the time dimension over spatial dimensions.'"}
        
        %% Phase 3: Time
        TemporalPass["3. Temporal Pass (Mamba Layer)\n'For every specific node, process its entire history sequence using Mamba SSM.'"]
        AN2(Add & LayerNorm)
        
        %% Phase 4: Restore
        PivotS{"4. Pivot Back (Restore)\n'Rearrange data back to the original (Batch, Time, Node) structure.'"}

        %% Connections within block
        GAT_Input --> SpatialPass
        SpatialPass --> AN1
        AN1 --> PivotT
        PivotT --> TemporalPass
        TemporalPass --> AN2
        AN2 --> PivotS
    end

    %% --- Output Level ---
    SelectLast["Select Last Timestep State\n(We only need the most recent state to predict the future)"]
    OutputProj["Output Projection Head\n(Linear layers to transform hidden state into final predictions)"]
    Output(["End: Final Forecast\n(Predicted values over the horizon)"])

    %% Main Flow Connections
    Input --> InitProj
    InitProj --> GAT_Input
    PivotS --> SelectLast
    SelectLast --> OutputProj
    OutputProj --> Output

    %% Styling for clarity
    style ST_Block fill:#f4f4f9,stroke:#333,stroke-width:1px
    style SpatialPass fill:#d4e157,stroke:#333,color:black
    style TemporalPass fill:#4db6ac,stroke:#333,color:black
    style PivotT fill:#ffd54f,stroke:#333,color:black
    style PivotS fill:#ffd54f,stroke:#333,color:black



