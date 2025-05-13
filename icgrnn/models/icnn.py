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

from ..layers import ConvexLayer


class ConvexLayerICNN(nn.Module):
    """
    Input-Convex Neural Network constructed using ConvexLayer modules
    """
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super(ConvexLayerICNN, self).__init__()
        self.input_dim = input_dim

        # Build primary layers (no bias, weights must be non-negative)
        self.primary_layers = nn.ModuleList()

        # Build skip connections (with bias, no constraints on weights)
        self.skip_layers = nn.ModuleList()

        # Input dimension to first hidden dimension
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Primary path (using ConvexLayer to ensure non-negative weights)
            self.primary_layers.append(ConvexLayer(prev_dim, hidden_dim))

            # Skip connection from input (regular Linear layer)
            self.skip_layers.append(nn.Linear(input_dim, hidden_dim))

            prev_dim = hidden_dim

        # Output layer
        self.output_primary = ConvexLayer(prev_dim, output_dim)
        self.output_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Store original input for skip connections
        x0 = x

        # First hidden layer
        z = F.relu(self.primary_layers[0](x) + self.skip_layers[0](x0))

        # Subsequent hidden layers
        for i in range(1, len(self.primary_layers)):
            z = F.relu(self.primary_layers[i](z) + self.skip_layers[i](x0))

        # Output layer
        out = self.output_primary(z) + self.output_skip(x0)

        return out
