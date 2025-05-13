import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import MessagePassing
from sklearn.metrics import accuracy_score, confusion_matrix



class ConvexGCNLayer_pubmed(MessagePassing):
    """Graph Convolutional Layer with convexity constraints"""
    def __init__(self, in_channels, out_channels):
        super(ConvexGCNLayer_pubmed, self).__init__(aggr='add')
        # Linear transformation for source nodes (weights constrained to be non-negative)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        # Bias term
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Skip connection transformation when dimensions don't match
        self.skip_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights to be small positive values (for stability)
        nn.init.uniform_(self.weight, 0.0, 0.1)
        nn.init.zeros_(self.bias)
        if self.skip_proj is not None:
            nn.init.xavier_uniform_(self.skip_proj.weight)
            nn.init.zeros_(self.skip_proj.bias)

    def forward(self, x, edge_index, edge_weight=None):
        # Store original input for skip connection
        original_x = x

        # Apply normalization like in standard GCN
        row, col = edge_index
        deg = torch.bincount(row)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages
        out = self.propagate(edge_index, x=x, norm=norm)

        # Apply skip connection with proper projection if needed
        if self.skip_proj is not None:
            skip = self.skip_proj(original_x)
        else:
            skip = original_x

        # Return the combination
        return out + skip

    def message(self, x_j, norm):
        # Apply normalization
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # Apply convex transformation with non-negative weights
        weight = torch.clamp(self.weight, min=0.0)

        # Matrix multiplication with non-negative weights to ensure convexity
        return F.linear(aggr_out, weight.t(), self.bias)
