
from models import ConvexLayerICNN
from message_passing import ConvexMessagePassing
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

from layers import ConvexLayer

class ICGRNN(nn.Module):
    """Input Convex Graph Recurrent Neural Network using ConvexLayerICNN."""
    def __init__(self, input_dim, hidden_dim, output_dim, icnn_hidden_dims=[32, 32]):
        super(ICGRNN, self).__init__()
        self.hidden_dim = hidden_dim

        # Input-to-hidden convex transformation
        self.ih = ConvexLayerICNN(input_dim, icnn_hidden_dims, hidden_dim)

        # Hidden-to-hidden convex transformation
        self.hh = ConvexLayerICNN(hidden_dim, icnn_hidden_dims, hidden_dim)

        # Message passing layer
        self.message_passing = ConvexMessagePassing(hidden_dim, icnn_hidden_dims, hidden_dim)

        # Output layer
        self.output_layer = ConvexLayerICNN(hidden_dim, icnn_hidden_dims, output_dim)

    def forward(self, x, edge_index, h=None, steps=1):
        num_nodes = x.size(0) if x.dim() == 2 else x.size(1)

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(num_nodes, self.hidden_dim, device=x.device)

        outputs = []

        for t in range(steps):
            # Apply input transformation

            # print(x.shape) ##########################################################################################
            input_transform = self.ih(x)

            # Apply hidden transformation
            hidden_transform = self.hh(h)

            # Combine and apply activation
            combined = input_transform + hidden_transform
            h_tilde = F.relu(combined)  # ReLU maintains convexity

            # Apply message passing
            h = self.message_passing(h_tilde, edge_index)

            # Compute output for this time step
            out = self.output_layer(h)
            outputs.append(out)

        # Stack outputs for all time steps
        return torch.stack(outputs, dim=1), h

