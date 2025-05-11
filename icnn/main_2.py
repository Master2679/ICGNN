import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm

class ICGNNLayer(nn.Module):
    """
    Implementation of an Input Convex Graph Neural Network Layer
    as described in the paper.
    """
    def __init__(self, in_features, out_features, input_dim=None, activation=F.relu):
        super(ICGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        # Weight matrix W for node features (non-negative)
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        
        # Weight matrix A for input features (if provided)
        if input_dim is not None:
            self.has_input = True
            self.A = nn.Parameter(torch.Tensor(input_dim, out_features))
        else:
            self.has_input = False
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights
        nn.init.xavier_uniform_(self.W)
        if self.has_input:
            nn.init.xavier_uniform_(self.A)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None, u=None):
        # Ensure W is non-negative to enforce convexity
        W_non_neg = F.softplus(self.W)
        
        # Message passing with adjacency matrix
        # For simplicity, we're using a basic aggregation method here
        # In practice, you might want to use the PyTorch Geometric library for more efficient graph operations
        rows, cols = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        
        # Aggregate messages from neighbors
        output = torch.zeros(x.size(0), self.out_features, device=x.device)
        for i in range(edge_index.shape[1]):
            src, dst = rows[i], cols[i]
            output[dst] += edge_weight[i] * (x[src] @ W_non_neg)
        
        # Add input contribution if applicable
        if self.has_input and u is not None:
            # Ensure A is non-negative for input convexity
            A_non_neg = F.softplus(self.A)
            output += u @ A_non_neg
        
        # Add bias and apply activation
        output += self.bias
        return self.activation(output)

class ICGNN(nn.Module):
    """
    Full Input Convex Graph Neural Network model.
    """
    def __init__(self, node_dim, hidden_dims, input_dim=None, activation=F.relu, final_activation=None):
        super(ICGNN, self).__init__()
        
        self.num_layers = len(hidden_dims)
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(ICGNNLayer(node_dim, hidden_dims[0], input_dim, activation))
        
        # Hidden layers
        for i in range(1, self.num_layers):
            self.layers.append(ICGNNLayer(hidden_dims[i-1], hidden_dims[i], input_dim, activation))
        
        # Output layer
        self.final_activation = final_activation
    
    def forward(self, x, edge_index, edge_weight=None, u=None):
        for i in range(self.num_layers):
            if i == 0:
                h = self.layers[i](x, edge_index, edge_weight, u)
            else:
                # Residual connection to input u in each layer to ensure input convexity
                h = self.layers[i](h, edge_index, edge_weight, u)
        
        if self.final_activation is not None:
            h = self.final_activation(h)
        
        return h

def create_grid_graph(n):
    """
    Create an n×n grid graph representing a 2D grid domain.
    Returns graph G and node positions.
    """
    G = nx.grid_2d_graph(n, n)
    
    # Convert to node indices
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Store positions for visualization
    pos = {mapping[node]: node for node in list(nx.grid_2d_graph(n, n).nodes())}
    
    # Create edge index for PyTorch
    edges = list(G.edges())
    edge_index = torch.tensor([[e[0], e[1]] for e in edges] + [[e[1], e[0]] for e in edges]).t()
    
    return G, pos, edge_index

def load_optimal_control_data(n_grid=10, n_samples=1000):
    """
    Generate synthetic data for the optimal control problem.
    Here we're simulating a simple PDE problem on a grid.
    """
    # Create grid graph
    _, _, edge_index = create_grid_graph(n_grid)
    n_nodes = n_grid * n_grid
    
    # Synthetic data generation
    # For demonstration, we'll create a simple function that maps inputs to outputs
    X = torch.rand(n_samples, n_nodes, 1)  # Random input conditions
    
    # Generate target outputs (simulating solution to control problem)
    # In a real implementation, this would be replaced by solving the actual PDE
    Y = torch.zeros(n_samples, n_nodes, 1)
    for i in range(n_samples):
        # Simple diffusion-like process on the grid
        u = X[i].squeeze()
        y = torch.zeros(n_nodes)
        for _ in range(5):  # Simulate a few steps of diffusion
            for j in range(edge_index.shape[1]):
                src, dst = edge_index[0, j], edge_index[1, j]
                y[dst] += 0.1 * (u[src] - u[dst])
        Y[i] = y.unsqueeze(1)
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]
    
    return X_train, Y_train, X_test, Y_test, edge_index

def train_icgnn(model, X_train, Y_train, edge_index, epochs=100, lr=0.01):
    """
    Train the ICGNN model on the given data.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # In a real implementation, you would use batches here
        for i in range(len(X_train)):
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X_train[i].squeeze(-1), edge_index)
            loss = F.mse_loss(pred.unsqueeze(-1), Y_train[i])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    return losses

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load/generate data
    X_train, Y_train, X_test, Y_test, edge_index = load_optimal_control_data()
    
    # Initialize model
    node_features = 1  # For simplicity, using scalar node features
    hidden_dims = [32, 32, 16, 1]  # Hidden layer dimensions
    model = ICGNN(node_features, hidden_dims)
    
    # Train model
    losses = train_icgnn(model, X_train, Y_train, edge_index, epochs=100)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_losses = []
        for i in range(len(X_test)):
            pred = model(X_test[i].squeeze(-1), edge_index)
            loss = F.mse_loss(pred.unsqueeze(-1), Y_test[i])
            test_losses.append(loss.item())
        
        avg_test_loss = sum(test_losses) / len(test_losses)
        print(f"Average test loss: {avg_test_loss:.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # Save the model
    torch.save(model.state_dict(), 'icgnn_model.pth')
    
    print("Training and evaluation completed.")

if __name__ == "__main__":
    print(load_optimal_control_data())