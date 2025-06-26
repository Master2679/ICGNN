import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch_geometric.utils import to_networkx
import torch
import seaborn as sns

def plot_accuracy_pubmed(model, data):
    """Plot overall accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    # Calculate accuracies
    train_acc = accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu())
    val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
    test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Dataset splits
    splits = ['Train', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]

    # Create bar plot
    bars = plt.bar(splits, accuracies, color=['blue', 'green', 'red'], alpha=0.7)

    # Add a horizontal line for random guessing (1/3 for PubMed)
    plt.axhline(y=1/3, color='gray', linestyle='--',
                label=f'Random Guess: {1/3:.2f}')

    # Customize plot
    plt.ylim(0, 1.0)
    plt.title('ICGCN Model Accuracy on PubMed Dataset', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('icgcn_pubmed_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f'Train accuracy: {train_acc:.4f}')
    print(f'Validation accuracy: {val_acc:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')


def plot_confusion_matrix_pubmed(model, data):
    """Plot confusion matrix with numeric class labels"""
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    # Extract test predictions and labels
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Get number of classes
    num_classes = int(data.y.max().item() + 1)
    class_labels = [str(i+1) for i in range(num_classes)]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))

    # Create heatmap with raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)

    plt.title('Confusion Matrix for ICGCN on PubMed Test Set', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)

    plt.tight_layout()
    plt.savefig('icgcn_pubmed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_accuracy_pubmed(model, data):
    """Plot class-wise accuracy with numeric labels"""
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
        if np.sum(class_mask) > 0:
            # Accuracy for this class
            class_acc[i] = np.sum((y_pred == i) & class_mask) / np.sum(class_mask)

    # Create numeric class labels
    class_labels = [str(i+1) for i in range(num_classes)]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Create bar chart
    bars = plt.bar(class_labels, class_acc, color='teal', alpha=0.7)

    # Add a line for the mean accuracy
    mean_acc = np.mean(class_acc)
    plt.axhline(y=mean_acc, color='red', linestyle='--',
                label=f'Mean Accuracy: {mean_acc:.3f}')

    # Customize plot
    plt.ylim(0, 1.0)
    plt.title('Class-wise Accuracy of ICGCN on PubMed Test Set', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()

    # Add count of samples per class
    class_counts = [np.sum(y_true == i) for i in range(num_classes)]

    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}\n(n={class_counts[i]})',
                ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('icgcn_pubmed_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print results
    print("Class-wise accuracy:")
    for i in range(num_classes):
        print(f"Class {i+1}: {class_acc[i]:.4f} (n={class_counts[i]})")
    print(f"Mean accuracy: {mean_acc:.4f}")

