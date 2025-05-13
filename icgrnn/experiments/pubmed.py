
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

from ..models import Pubmed_Node_Classifier
from ..helper_scripts import train_icgcn_pubmed
from ..visualization import plot_accuracy_pubmed,plot_confusion_matrix_pubmed,plot_class_accuracy_pubmed

# Load PubMed dataset
dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=NormalizeFeatures())
data = dataset[0]  # Get the first and only graph

print(f"\nPubMed Dataset:")
print(f"  Number of nodes: {data.x.size(0)}")
print(f"  Number of edges: {data.edge_index.size(1)//2}")
print(f"  Number of features: {data.x.size(1)}")
print(f"  Number of classes: {dataset.num_classes}")
print(f"  Number of training nodes: {data.train_mask.sum().item()}")
print(f"  Number of validation nodes: {data.val_mask.sum().item()}")
print(f"  Number of test nodes: {data.test_mask.sum().item()}\n")

# Initialize model - tuned for PubMed
model = Pubmed_Node_Classifier(
    in_channels=data.num_features,
    hidden_channels=32,
    out_channels=dataset.num_classes,
    num_layers=1,  # Deeper model for PubMed
    dropout=0.9
)

print("Training Input Convex GCN model on PubMed dataset...")
model, train_accs, val_accs, test_accs = train_icgcn_pubmed(
    model, data, num_epochs=2000, lr=0.001, weight_decay=1e-4, patience=30
)

# Plot training curve
plt.figure(figsize=(10, 6))
epochs = range(10, 10*(len(train_accs)+1), 10)
plt.plot(epochs, train_accs, 'b-', label='Train')
plt.plot(epochs, val_accs, 'g-', label='Validation')
plt.plot(epochs, test_accs, 'r-', label='Test')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('ICGCN Training Progress on PubMed', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('icgcn_pubmed_training_curve.png', dpi=300)
plt.show()

# Plot overall accuracy
plot_accuracy_pubmed(model, data)

# Plot confusion matrix
plot_confusion_matrix_pubmed(model, data)

# Plot class-wise accuracy
plot_class_accuracy_pubmed(model, data)