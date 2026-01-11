📊 ST-Mamba for Traffic Forecasting

This repository contains an implementation of ST-Mamba for spatio-temporal traffic forecasting on graph-structured data (e.g., METR-LA, PeMS).
✅ Project Checklist (End-to-End)
1️⃣ Problem Definition

Task: Multivariate spatio-temporal time-series forecasting

Input shape: (B, T_in, N, F)

Output shape: (B, T_out, N)

Prediction target:

Speed

Flow

    Occupancy

    Forecast horizons: 15 / 30 / 60 minutes

2️⃣ Dataset & Preprocessing

Load raw traffic dataset

Handle missing values (zero mask / interpolation)

Normalize data (mean & std)

Save normalization statistics

Create sliding windows:

Past window (T_in)

    Future window (T_out)

Split dataset:

Train

Validation

    Test

    Create PyTorch Dataset & DataLoader

3️⃣ Graph Construction

Load adjacency matrix

Distance-based or correlation-based edges

Normalize adjacency (if needed)

    Verify node count matches dataset

4️⃣ Model (ST-Mamba)

Spatial module implemented

Temporal Mamba module implemented

Input projection layer

Output projection layer

Shape consistency check

    Causal temporal processing (no future leakage)

5️⃣ Loss Function

MAE (default)

Optional: MAE + MSE hybrid

    Masked loss for missing sensors

loss_fn = torch.nn.L1Loss()

6️⃣ Optimizer & Scheduler

AdamW optimizer

Weight decay enabled

    Learning rate scheduler

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, patience=5)

7️⃣ Training Loop

Forward pass

Loss computation

Backpropagation

Gradient clipping

Optimizer step

Scheduler step

    Epoch-wise logging

8️⃣ Validation & Testing

Separate validation loop

Metrics:

MAE

RMSE

    MAPE

De-normalize predictions before evaluation

    Horizon-wise reporting (15 / 30 / 60 min)

9️⃣ Debugging & Sanity Checks

Overfit on a single batch

Check tensor shapes at each layer

Confirm loss decreases

    Check for NaNs / exploding gradients

🔟 Experiment Management

Save best checkpoints

Log training curves

Fix random seed

    Store hyperparameters

1️⃣1️⃣ Final Results

Test set evaluation

Compare with baselines

Visualize predictions vs ground truth

    Document results in README

