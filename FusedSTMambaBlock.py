class FusedSTMambaBlock(nn.Module):
    """
    True ST-Mamba: GAT and Mamba logic fused into a single class.
    """
    def __init__(self, d_model, d_state=16, num_heads=4, dropout=0.1):
        super().__init__()
        
        # --- PART A: The Spatial Tools (GAT Parameters) ---
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # We define the GAT weights directly here
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)
        
        # --- PART B: The Temporal Tools (Mamba Parameters) ---
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=4, expand=2)
        
        # --- PART C: The Glue (Norms & Dropout) ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        B, T, N, F = x.shape
        
        # ==========================================
        # STEP 1: SPATIAL MIXING (GAT LOGIC INLINE)
        # ==========================================
        # Flatten Batch & Time
        residual = x.view(B*T, N, F)
        h = residual
        
        # 1.1 Linear Projections for Attention
        q = self.W_q(h).view(B*T, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(h).view(B*T, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(h).view(B*T, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 1.2 Calculate Attention Scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 1.3 Apply Graph Mask (Using Adjacency Matrix)
        mask = (adj == 0).view(1, 1, N, N)
        scores = scores.masked_fill(mask, -1e9)
        
        # 1.4 Softmax & Aggregation
        attn = F.softmax(scores, dim=-1)
        out_spatial = torch.matmul(attn, v)
        
        # 1.5 Merge Heads
        out_spatial = out_spatial.transpose(1, 2).reshape(B*T, N, F)
        out_spatial = self.W_out(out_spatial)
        
        # 1.6 Spatial Residual Connection
        x_spatial = self.norm1(residual + self.dropout(out_spatial))
        
        # ==========================================
        # STEP 2: THE PIVOT (Rotate for Mamba)
        # ==========================================
        # [B*T, N, F] -> [B, T, N, F] -> [B, N, T, F] -> [B*N, T, F]
        x_mamba_in = x_spatial.view(B, T, N, F).permute(0, 2, 1, 3).reshape(B*N, T, F)
        
        # ==========================================
        # STEP 3: TEMPORAL MIXING (MAMBA LOGIC)
        # ==========================================
        x_mamba_out = self.mamba(x_mamba_in)
        
        # Temporal Residual Connection
        x_temporal = self.norm2(x_mamba_in + self.dropout(x_mamba_out))
        
        # ==========================================
        # STEP 4: RESTORE SHAPE
        # ==========================================
        # [B*N, T, F] -> [B, N, T, F] -> [B, T, N, F]
        x_final = x_temporal.view(B, N, T, F).permute(0, 2, 1, 3).contiguous()
        
        return x_final
