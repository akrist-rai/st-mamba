### 3. Head Merging (Re-Unification)
After the parallel attention computations, the independent heads are re-assembled into a single feature stream.

1.  **Transpose:** Re-aligns the tensor so that all head outputs for a specific node are adjacent.
2.  **Reshape (Concatenation):** Fuses the independent head vectors (`Heads × HeadDim`) back into the original feature dimension (`d_model`).
3.  **Linear Projection (`W_out`):** A final mixing layer that blends the information from different heads. This ensures that the distinct insights (e.g., speed vs. weather) are integrated into a unified representation before passing to the next block.
