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


class ConvexLayer(nn.Module):
    """A convex layer that ensures convexity with respect to inputs."""
    def __init__(self, in_dim, out_dim):
        super(ConvexLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights to be non-negative to ensure convexity
        nn.init.uniform_(self.weight, 0.0, 0.1)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Using ReLU to maintain non-negativity of weights
        return F.linear(x, F.relu(self.weight), self.bias)
