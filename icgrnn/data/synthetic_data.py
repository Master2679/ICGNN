import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data


def create_heat_diffusion_dataset(num_nodes=10, num_timesteps=20, num_samples=100, diffusion_rate=0.1):
    """
    Create synthetic data for heat diffusion with control inputs on a graph.

    Parameters:
    - num_nodes: Number of nodes in the graph
    - num_timesteps: Number of time steps to simulate
    - num_samples: Number of different initial conditions to generate
    - diffusion_rate: Rate of heat diffusion between connected nodes

    Returns:
    - graph_data: PyG Data object with the graph structure
    - features: Input features including initial temperatures and control signals
    - targets: Optimal temperature evolution under controlled diffusion
    """
    # Create a random connected graph
    # Using a cycle graph with some random edges for better connectivity
    G = nx.cycle_graph(num_nodes)

    # Add some random edges to make it more interesting
    for _ in range(num_nodes):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            G.add_edge(i, j)

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Add reverse edges to make it undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Create adjacency matrix for simulation
    adj_matrix = nx.to_numpy_array(G)

    # Initialize dataset containers
    features_list = []
    targets_list = []

    for sample in range(num_samples):
        # Initial temperature (random between 0 and 1)
        initial_temp = torch.rand(num_nodes, 1)

        # Target temperature (random between 0 and 1)
        target_temp = torch.rand(num_nodes, 1)

        # Initialize temperature trajectory and control inputs
        temp_trajectory = torch.zeros(num_nodes, num_timesteps)
        control_inputs = torch.zeros(num_nodes, num_timesteps)
        temp_trajectory[:, 0] = initial_temp.squeeze()

        # Simulate heat diffusion with simple control
        for t in range(1, num_timesteps):
            # Current temperature
            current_temp = temp_trajectory[:, t-1].numpy()

            # Compute diffusion (simplified heat equation on graph)
            diffusion = diffusion_rate * (adj_matrix @ current_temp - current_temp * np.sum(adj_matrix, axis=1))

            # Compute control input (simplified proportional control towards target)
            error = target_temp.numpy().squeeze() - current_temp
            control = 0.05 * error  # Proportional control factor

            # Apply diffusion and control
            next_temp = current_temp + diffusion + control

            # Store values
            temp_trajectory[:, t] = torch.tensor(next_temp)
            control_inputs[:, t-1] = torch.tensor(control)

        # Create input features: [initial_temp, target_temp]
        node_features = torch.cat([initial_temp, target_temp], dim=1)

        # Store this sample
        features_list.append(node_features)
        targets_list.append(torch.cat([temp_trajectory, control_inputs], dim=1))

    # Stack all samples
    features = torch.stack(features_list)  # [num_samples, num_nodes, 2]
    targets = torch.stack(targets_list)    # [num_samples, num_nodes, 2*num_timesteps]

    # Create PyG data object
    graph_data = Data(edge_index=edge_index)

    return graph_data, features, targets