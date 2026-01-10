
python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------------------------------------
# OPTIONAL: Import Mamba (with CPU fallback for testing)
# --------------------------------------------------------
try:
    from mamba_ssm import Mamba
except ImportError:
    print("WARNING: mamba_ssm not found. Using a lightweight CPU placeholder for testing.")
    class Mamba(nn.Module):
        """Placeholder for systems without CUDA/Mamba installed"""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.in_proj = nn.Linear(d_model, d_model * 2)
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv//2)
            self.out_proj = nn.Linear(d_model, d_model)
        def forward(self, x):
            # x: [Batch, Time, Dim]
            return self.out_proj(F.silu(self.in_proj(x))[:, :, :x.shape[-1]])

# ==========================================
# 1. THE SPATIAL SPECIALIST (GAT LAYER)
# ==========================================
class GATLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer.
    Equation: Output = Softmax(Attention_Scores * Mask) * Values
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        # Learnable Linear Transforms
        self.W_query = nn.Linear(in_features, out_features, bias=False)
        self.W_key = nn.Linear(in_features, out_features, bias=False)
        self.W_value = nn.Linear(in_features, out_features, bias=False)
        
        # Output projection to mix heads
        self.out_proj = nn.Linear(out_features, out_features)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialization
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_key.weight)
        nn.init.xavier_uniform_(self.W_value.weight)

    def forward(self, h, adj):
        """
        Args:
            h: [Batch, N, In_Features] (Node embeddings)
            adj: [N, N] (Adjacency Matrix: 1 if connected, 0 else)
        """
        B, N, _ = h.shape
        
        # 1. Linear Projection & Split Heads
        # [B, N, Features] -> [B, N, Heads, HeadDim] -> [B, Heads, N, HeadDim]
        q = self.W_query(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Calculate Attention Scores (Scaled Dot Product)
        # [B, Heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 3. Apply Mask (The "Graph" Logic)
        # We assume adj is [N, N]. We reshape to broadcast: [1, 1, N, N]
        # Where adj == 0, set score to -infinity (so softmax becomes 0)
        mask = (adj == 0).view(1, 1, N, N)
        scores = scores.masked_fill(mask, -1e9)
        
        # 4. Softmax & Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. Aggregate Values
        # [B, Heads, N, N] @ [B, Heads, N, HeadDim] -> [B, Heads, N, HeadDim]
        out = torch.matmul(attn_weights, v)
        
        # 6. Merge Heads
        # [B, Heads, N, HeadDim] -> [B, N, Heads, HeadDim] -> [B, N, Features]
        out = out.transpose(1, 2).reshape(B, N, self.out_features)
        
        # 7. Final Projection + Residual Connection
        return h + self.out_proj(out)


# ==========================================
# 2. THE ENGINE BLOCK (ST-MAMBA)
# ==========================================
class STMambaBlock(nn.Module):
    """
    Interleaved Spatio-Temporal Block.
    Flow: Input -> GAT (Space) -> Pivot -> Mamba (Time) -> Pivot Back -> Output
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Spatial Layer (GAT)
        self.spatial_layer = GATLayer(d_model, d_model, num_heads=num_heads, dropout=dropout)
        self.norm_spatial = nn.LayerNorm(d_model)
        
        # Temporal Layer (Mamba)
        self.temporal_layer = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_temporal = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_matrix):
        B, T, N, F = x.shape
        
        # --- PHASE 1: SPATIAL (GAT) ---
        # View: [Batch * Time, Nodes, Features]
        x_slices = x.view(B * T, N, F)
        
        x_spatial = self.spatial_layer(x_slices, adj_matrix)
        x_spatial = self.norm_spatial(x_slices + self.dropout(x_spatial))
        
        # Restore: [Batch, Time, Nodes, Features]
        x = x_spatial.view(B, T, N, F)
        
        
        # --- PHASE 2: PIVOT (Rotate to Time) ---
        # Swap Time & Nodes -> [Batch, Nodes, Time, Features]
        x_rotated = x.permute(0, 2, 1, 3)
        
        # Reshape to flatten Batch & Nodes -> [Batch * Nodes, Time, Features]
        # This makes the memory contiguous for Mamba
        x_temporal_input = x_rotated.reshape(B * N, T, F)
        
        
        # --- PHASE 3: TEMPORAL (Mamba) ---
        x_temporal_out = self.temporal_layer(x_temporal_input)
        x_temporal = self.norm_temporal(x_temporal_input + self.dropout(x_temporal_out))
        
        
        # --- PHASE 4: RESTORE (Rotate Back) ---
        # Unflatten -> [Batch, Nodes, Time, Features]
        x_restored = x_temporal.view(B, N, T, F)
        
        # Swap back -> [Batch, Time, Nodes, Features]
        x_final = x_restored.permute(0, 2, 1, 3).contiguous()
        
        return x_final


# ==========================================
# 3. THE MAIN MODEL (WRAPPER)
# ==========================================
class NewtonGraphMamba(nn.Module):
    def __init__(self, 
                 in_features=5, 
                 d_model=64, 
                 num_nodes=100,
                 d_state=16, 
                 num_layers=4, 
                 num_heads=4,
                 prediction_horizon=12):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Input Projection: Raw Features -> Hidden Dim
        self.input_proj = nn.Linear(in_features, d_model)
        
        # Stack of ST-Mamba Blocks
        self.layers = nn.ModuleList([
            STMambaBlock(d_model, d_state, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Output Projection: Hidden Dim -> Predictions
        # We predict 1 value (e.g., speed) for the next 'prediction_horizon' steps
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, prediction_horizon) 
        )

    def forward(self, x, adj_matrix):
        """
        x: [Batch, History_Len, Nodes, Features]
        adj: [Nodes, Nodes]
        """
        # 1. Project Input
        x = self.input_proj(x) # [B, T, N, 64]
        
        # 2. Pass through Layers
        for layer in self.layers:
            x = layer(x, adj_matrix)
            
        # 3. Prediction Head
        # We take the state at the LAST timestamp (T) to predict T+1...T+12
        last_state = x[:, -1, :, :] # [B, N, 64]
        
        # [B, N, Horizon]
        prediction = self.output_proj(last_state)
        
        return prediction

# ==========================================
# 4. TESTING CODE (Run this to verify)
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 2
    HISTORY_LEN = 12
    NUM_NODES = 20
    FEATURES = 5
    D_MODEL = 32
    HORIZON = 6
    
    # Create Dummy Data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    # Random Traffic Data [Batch, Time, Nodes, Features]
    x = torch.randn(BATCH_SIZE, HISTORY_LEN, NUM_NODES, FEATURES).to(device)
    
    # Random Adjacency Matrix (Binary)
    adj = torch.randint(0, 2, (NUM_NODES, NUM_NODES)).float().to(device)
    # Ensure diagonal is 1 (self-loops)
    adj.fill_diagonal_(1)
    
    # Initialize Model
    model = NewtonGraphMamba(
        in_features=FEATURES,
        d_model=D_MODEL,
        num_nodes=NUM_NODES,
        num_layers=2,
        prediction_horizon=HORIZON
    ).to(device)
    
    # Forward Pass
    try:
        y_pred = model(x, adj)
        print("\n✅ Model Forward Pass Successful!")
        print(f"Input Shape: {x.shape}")
        print(f"Output Shape: {y_pred.shape}")
        print(f"Expected:    torch.Size([{BATCH_SIZE}, {NUM_NODES}, {HORIZON}])")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
