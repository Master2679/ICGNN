from message_passing import ConvexMessagePassing
from models import ConvexLayerICNN

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



class HeatDiffusionICGRNN(nn.Module):
    """Input Convex Graph RNN for heat diffusion control with stability improvements."""
    def __init__(self, input_dim, hidden_dim, output_dim=1, icnn_hidden_dims=[32, 32]):
        super(HeatDiffusionICGRNN, self).__init__()
        self.hidden_dim = hidden_dim

        # Input-to-hidden convex transformation
        self.ih = ConvexLayerICNN(input_dim, icnn_hidden_dims, hidden_dim)

        # Hidden-to-hidden convex transformation
        self.hh = ConvexLayerICNN(hidden_dim, icnn_hidden_dims, hidden_dim)

        # Message passing layer
        self.message_passing = ConvexMessagePassing(hidden_dim, icnn_hidden_dims, hidden_dim)

        # Output layers - separated for temperature and control
        self.temp_output = ConvexLayerICNN(hidden_dim, icnn_hidden_dims, output_dim)
        self.control_output = ConvexLayerICNN(hidden_dim, icnn_hidden_dims, output_dim)

        # Add normalization layers for stability
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.pre_temp_norm = nn.LayerNorm(hidden_dim)
        self.pre_control_norm = nn.LayerNorm(hidden_dim)

        # Initialize scaling factor for stability
        self.scale_factor = 0.1

    def forward(self, x, edge_index, h=None, steps=20):
        batch_size = 1 if x.dim() == 2 else x.size(0)
        num_nodes = x.size(-2)
        # print(batch_size)

        # Reshape if we have a batch dimension
        if x.dim() == 3:  # [batch, nodes, features]
            x_flat = x.reshape(-1, x.size(-1))
        else:  # [nodes, features]
            x_flat = x

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size * num_nodes, self.hidden_dim, device=x.device)

        temp_outputs = []
        control_outputs = []

        for t in range(steps):
            # Apply input with scaling for stability
            input_transform = self.scale_factor * self.ih(x_flat)

            # Apply hidden transformation with scaling for stability
            hidden_transform = self.scale_factor * self.hh(h)

            # Combine and apply activation
            combined = input_transform + hidden_transform
            h_tilde = F.relu(combined)  # ReLU maintains convexity

            # Apply message passing
            if batch_size > 1:
                # This is a simplification - in practice you'd use PyG's batch handling
                batched_edge_index = []
                for b in range(batch_size):
                    offset = b * num_nodes
                    batched_edge_index.append(edge_index + offset)
                batched_edge_index = torch.cat(batched_edge_index, dim=1)
                h_raw = self.message_passing(h_tilde, batched_edge_index)
            else:
                h_raw = self.message_passing(h_tilde, edge_index)
                # h_raw = self.message_passing(h_raw, edge_index)
                # h_raw = self.message_passing(h_raw, edge_index)

            # Apply normalization to hidden state for stability
            h = self.hidden_norm(h_raw)

            # Compute temperature output with normalization and clamping
            temp_out = self.temp_output(self.pre_temp_norm(h))
            # Clamp temperature values to reasonable range
            temp_out = torch.clamp(temp_out, min=0.0, max=100.0)
            # print(temp_out)
            # print("TempOut")
            temp_outputs.append(temp_out)

            # Compute control output with normalization and clamping
            control_out = self.control_output(self.pre_control_norm(h))
            # Clamp control values to reasonable range
            control_out = torch.clamp(control_out, min=-50.0, max=50.0)
            control_outputs.append(control_out)

        # Stack outputs for all time steps and reshape
        temp_tensor = torch.stack(temp_outputs, dim=1)  # [batch*nodes, steps, 1]
        control_tensor = torch.stack(control_outputs, dim=1)  # [batch*nodes, steps, 1]

        # Reshape to [batch, nodes, steps, 1] if we have a batch
        if batch_size > 1:
            temp_tensor = temp_tensor.view(batch_size, num_nodes, steps, 1)
            control_tensor = control_tensor.view(batch_size, num_nodes, steps, 1)
        else:
            temp_tensor = temp_tensor.view(num_nodes, steps, 1)
            control_tensor = control_tensor.view(num_nodes, steps, 1)

        # Squeeze the last dimension
        temp_tensor = temp_tensor.squeeze(-1)
        control_tensor = control_tensor.squeeze(-1)

        return temp_tensor, control_tensor, h