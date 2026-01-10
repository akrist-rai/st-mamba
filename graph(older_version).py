class NewtonGraphMamba(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 in_features=5, 
                 d_model=64, 
                 d_state=16, 
                 num_layers=4, 
                 prediction_horizon=12):
        super().__init__()
        
        # 1. Input Projection (The Entry Gate)
        # Converts raw data (Speed, Rain) -> Rich Embeddings
        self.input_proj = nn.Linear(in_features, d_model)
        
        # 2. The Engine (Stack of ST-Mamba Blocks)
        # We stack the blocks to learn complex patterns
        self.layers = nn.ModuleList([
            STMambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(num_layers)
        ])
        
        # 3. Output Projection (The Exit Gate)
        # Converts Rich Embeddings -> Actual Predictions (Next 12 speeds)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, prediction_horizon)
        )

    def forward(self, x, adj_matrix):
        """
        x: [Batch, History_Len, Nodes, Features]
        adj_matrix: [Nodes, Nodes]
        """
        # Step 1: Project Raw Features
        # [B, T, N, 5] -> [B, T, N, 64]
        x = self.input_proj(x)
        
        # Step 2: Pass through the Interleaved Blocks
        for layer in self.layers:
            x = layer(x, adj_matrix)
            
        # Step 3: Predict Future
        # We typically take the LAST time step of the sequence to predict the future
        # [B, T, N, 64] -> [B, N, 64] (Take last T)
        last_state = x[:, -1, :, :]
        
        # [B, N, 64] -> [B, N, 12] (Predict next 12 steps)
        prediction = self.output_proj(last_state)
        
        return prediction
