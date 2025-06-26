
from layers import ConvexGCNLayer_pubmed


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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

class Pubmed_Node_Classifier(nn.Module):
    """Input Convex Graph Convolutional Network with regularization"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(Pubmed_Node_Classifier, self).__init__()

        self.dropout = dropout

        # Feature dimensionality reduction to combat overfitting
        self.feature_proj = nn.Linear(in_channels, hidden_channels)

        # Input layer
        self.conv_first = ConvexGCNLayer_pubmed(hidden_channels, hidden_channels)

        # Hidden layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(ConvexGCNLayer_pubmed(hidden_channels, hidden_channels))

        # Output layer
        self.classifier = nn.Linear(hidden_channels, out_channels)

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index):
        # Project features to lower dimension
        x = self.feature_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # First layer
        x = self.conv_first(x, edge_index)
        x = self.layer_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hidden layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.layer_norms[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.classifier(x)

        return x