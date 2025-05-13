
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

def train_icgrnn(model, graph_data, target, num_epochs=100, lr=0.01):
    """
    Train the ICGRNN model with gradient clipping and diagnostics.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # If target is for the final step only
    final_step_only = (len(target.shape) == 2)

    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with diagnostic checks
        try:
            output, hidden = model(graph_data.x, graph_data.edge_index,
                                  steps=target.shape[1] if not final_step_only else 1)

            # Check for NaN in forward pass
            if torch.isnan(output).any():
                print(f"Epoch {epoch+1}: NaN in output detected")
                # We could try to identify which component is generating NaNs here
                continue

            # Compute loss
            if final_step_only:
                # If target is only for the final time step
                loss = F.mse_loss(output[:, -1, :], target)
            else:
                # If target is for all time steps
                loss = F.mse_loss(output, target)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Epoch {epoch+1}: NaN loss detected, skipping update")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Print progress
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')

        except Exception as e:
            print(f"Epoch {epoch+1}: Error during training: {e}")
            continue

    return model