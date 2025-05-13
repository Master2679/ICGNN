import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

from ..models import CoraNodeClassifier
from ..helper_scripts import train_icgcn,evaluate_icgcn
from ..visualization import plot_class_accuracy


# Set random seed for reproducibility
torch.manual_seed(42)

# Load CORA dataset
dataset = Planetoid(root='/tmp/CORA', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Get the first and only graph

print(f"\nCORA Dataset:")
print(f"  Number of nodes: {data.x.size(0)}")
print(f"  Number of edges: {data.edge_index.size(1)//2}")  # Divide by 2 for undirected edges
print(f"  Number of features: {data.x.size(1)}")
print(f"  Number of classes: {dataset.num_classes}")
print(f"  Number of training nodes: {data.train_mask.sum().item()}")
print(f"  Number of validation nodes: {data.val_mask.sum().item()}")
print(f"  Number of test nodes: {data.test_mask.sum().item()}\n")

# Initialize model
model = CoraNodeClassifier(
in_channels=data.num_features,
hidden_channels=32,  # Reduced from 64
out_channels=dataset.num_classes,
num_layers=2,
dropout=0.6  # Increased from 0.5
)

model = train_icgcn(model, data, num_epochs=3000, lr=0.001, weight_decay=1e-3, patience=30)
class_acc, class_counts = plot_class_accuracy(model, data)