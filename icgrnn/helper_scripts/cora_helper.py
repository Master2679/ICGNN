

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

def train_icgcn(model, data, num_epochs=200, lr=0.005, weight_decay=1e-3, patience=30):
    """Train with early stopping"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = None
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training mode
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits = model(data.x, data.edge_index)

        # Compute loss (only for training nodes)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

        # Add L1 regularization for sparsity
        l1_reg = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, 1)

        loss += 1e-5 * l1_reg

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Evaluation mode
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Forward pass
                logits = model(data.x, data.edge_index)
                pred = logits.argmax(dim=1)

                # Compute accuracies
                train_acc = accuracy_score(data.y[data.train_mask].cpu(),
                                          pred[data.train_mask].cpu())
                val_acc = accuracy_score(data.y[data.val_mask].cpu(),
                                        pred[data.val_mask].cpu())
                test_acc = accuracy_score(data.y[data.test_mask].cpu(),
                                         pred[data.test_mask].cpu())

                print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}, '
                      f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

                # Early stopping logic
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model

def evaluate_icgcn(model, data):
    """Evaluate the trained model on CORA dataset"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

        # Compute metrics
        train_acc = accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu())
        val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
        test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

        test_f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(),
                         average='macro')

        print("\nFinal results:")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")

        return test_acc, test_f1, logits
    
