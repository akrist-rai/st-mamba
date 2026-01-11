# Loss Functions

This project involves **time-series forecasting on traffic data**, where the model predicts **continuous future values** (e.g., speed, flow, occupancy).  
Choosing the correct loss function is critical for stable training and meaningful evaluation.

---

## Problem Type

**Regression (not classification)**

- Model outputs are **real-valued tensors**
- Outputs do **not** represent probabilities
- Values have **physical meaning** (km/h, vehicles/hour, %)

Therefore, **regression loss functions** are used.

---

## Mean Absolute Error (MAE)

### Definition
\[
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
\]

### Interpretation
MAE measures the **average absolute difference** between predicted and true values.

Example:
- MAE = 3 → predictions are off by **3 units on average**

### Why MAE is used
- Robust to outliers
- Stable when sensor noise is present
- Directly interpretable in real-world units
- Standard metric in traffic benchmarks (METR-LA, PeMS)

### PyTorch implementation
```python
loss_fn = torch.nn.L1Loss()
