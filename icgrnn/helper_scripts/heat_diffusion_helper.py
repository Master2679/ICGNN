
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import networkx as nx

def train_heat_diffusion_model(model, graph_data, features, targets, num_epochs=100, lr=0.001, weight_decay=1e-5, patience=15):
    """
    Train the heat diffusion model with improved stability measures.

    Parameters:
    - model: HeatDiffusionICGRNN model
    - graph_data: PyG Data object with graph structure
    - features: Input features
    - targets: Target temperature and control trajectories
    - num_epochs: Maximum number of training epochs
    - lr: Learning rate
    - weight_decay: L2 regularization factor
    - patience: Early stopping patience
    """
    # Initialize optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5, verbose=True
    )

    num_samples = features.size(0)
    num_nodes = features.size(1)
    num_timesteps = targets.size(2) // 2

    # Separate temperature and control targets
    temp_targets = targets[:, :, :num_timesteps]  # [samples, nodes, timesteps]
    control_targets = targets[:, :, num_timesteps:]  # [samples, nodes, timesteps]

    # For early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_temp_loss = 0
        total_control_loss = 0

        # Process each sample
        for i in range(num_samples):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            temp_pred, control_pred, _ = model(features[i], graph_data.edge_index, steps=num_timesteps)

            # Compute losses
            temp_loss = F.mse_loss(temp_pred, temp_targets[i])
            control_loss = F.mse_loss(control_pred, control_targets[i])
            loss = temp_loss + 0.1 * control_loss  # Weighting factor for control loss

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}, sample {i}")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping (critical for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_temp_loss += temp_loss.item()
            total_control_loss += control_loss.item()

        # Calculate average losses
        avg_loss = total_loss / num_samples
        avg_temp_loss = total_temp_loss / num_samples
        avg_control_loss = total_control_loss / num_samples

        # Update learning rate based on average loss
        scheduler.step(avg_loss)

        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, '
                  f'Temp Loss: {avg_temp_loss:.4f}, Control Loss: {avg_control_loss:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                # Restore best model
                model.load_state_dict(best_model_state)
                break

    return model