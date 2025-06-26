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

from models import ConvexLayerICNN

class ConvexMessagePassing(MessagePassing):
    """Message passing layer that maintains input convexity."""
    def __init__(self, in_channels, hidden_dims, out_channels):
        super(ConvexMessagePassing, self).__init__(aggr='add')
        self.convex_transform = ConvexLayerICNN(in_channels, hidden_dims, out_channels)

    def forward(self, x, edge_index):
        # Propagate messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Transform messages in a convex manner
        return self.convex_transform(x_j)

    def update(self, aggr_out):
        # No additional transformation needed here
        return aggr_out