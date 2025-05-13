import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

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

def plot_class_accuracy(model, data):
    """
    Plot class-wise accuracy with simple numeric labels

    Parameters:
    - model: Trained ICGCN model
    - data: PyG Data object containing CORA dataset
    """
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    # Extract test predictions and labels
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

    # Calculate per-class accuracy
    num_classes = int(data.y.max().item() + 1)
    class_acc = np.zeros(num_classes)

    for i in range(num_classes):
        # Subset of test nodes with true class i
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:  # avoid division by zero
            # Accuracy for this class
            class_acc[i] = np.sum((y_pred == i) & class_mask) / np.sum(class_mask)

    # Create simple numeric class labels
    class_labels = [str(i+1) for i in range(num_classes)]

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Create bar chart
    bars = plt.bar(class_labels, class_acc, color='teal', alpha=0.7)

    # Add a line for the mean accuracy
    mean_acc = np.mean(class_acc)
    plt.axhline(y=mean_acc, color='red', linestyle='--',
                label=f'Mean Accuracy: {mean_acc:.3f}')

    # Customize plot
    plt.ylim(0, 1.0)
    plt.title('Class-wise Accuracy of ICGCN on CORA Test Set', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)

    # Add count of samples per class
    class_counts = [np.sum(y_true == i) for i in range(num_classes)]

    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}\n(n={class_counts[i]})',
                ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('icgcn_cora_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print results
    print("Class-wise accuracy:")
    for i in range(num_classes):
        print(f"Class {i+1}: {class_acc[i]:.4f} (n={class_counts[i]})")
    print(f"Mean accuracy: {mean_acc:.4f}")

    return class_acc, class_counts

