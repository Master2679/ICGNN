import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
import torch

def visualize_model_performance(model, graph_data, features, targets, sample_idx=0, node_indices=None):
    """
    Visualize the performance of the heat diffusion model.

    Parameters:
    - model: Trained HeatDiffusionICGRNN model
    - graph_data: PyG Data object with graph structure
    - features: Input features [num_samples, num_nodes, input_dim]
    - targets: Target temperature and control trajectories [num_samples, num_nodes, 2*num_timesteps]
    - sample_idx: Index of the sample to visualize
    - node_indices: List of node indices to visualize (None for random selection)
    """
    # Set model to evaluation mode
    model.eval()

    # Get dimensions
    num_timesteps = targets.size(2) // 2
    num_nodes = features.size(1)

    # Separate temperature and control targets
    temp_targets = targets[:, :, :num_timesteps]  # [samples, nodes, timesteps]
    control_targets = targets[:, :, num_timesteps:]  # [samples, nodes, timesteps]

    # If no specific nodes provided, choose a few random ones
    if node_indices is None:
        node_indices = np.random.choice(num_nodes, size=min(3, num_nodes), replace=False)

    # Generate predictions
    with torch.no_grad():
        temp_pred, control_pred, _ = model(features[sample_idx], graph_data.edge_index, steps=num_timesteps)

    # Convert to numpy for plotting
    temp_targets_np = temp_targets[sample_idx].cpu().numpy()
    control_targets_np = control_targets[sample_idx].cpu().numpy()
    temp_pred_np = temp_pred.cpu().numpy()
    control_pred_np = control_pred.cpu().numpy()

    # Calculate errors
    temp_mse = np.mean((temp_targets_np - temp_pred_np) ** 2)
    control_mse = np.mean((control_targets_np - control_pred_np) ** 2)

    # 1. Plot temperature and control trajectories for selected nodes
    plt.figure(figsize=(15, 10))

    # Initial conditions and targets from features
    initial_temp = features[sample_idx, :, 0].cpu().numpy()
    target_temp = features[sample_idx, :, 1].cpu().numpy()

    # Temperature trajectories
    plt.subplot(2, 2, 1)
    for i in node_indices:
        plt.plot(temp_targets_np[i], 'b-', label=f'Target (Node {i})' if i == node_indices[0] else "")
        plt.plot(temp_pred_np[i], 'r--', label=f'Prediction (Node {i})' if i == node_indices[0] else "")

    plt.title('Temperature Trajectories')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Control inputs
    plt.subplot(2, 2, 2)
    for i in node_indices:
        plt.plot(control_targets_np[i], 'g-', label=f'Target Control (Node {i})' if i == node_indices[0] else "")
        plt.plot(control_pred_np[i], 'm--', label=f'Predicted Control (Node {i})' if i == node_indices[0] else "")

    plt.title('Control Inputs')
    plt.xlabel('Time Step')
    plt.ylabel('Control Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Visualize error distributions
    plt.subplot(2, 2, 3)

    # Temperature error distribution across all nodes
    temp_errors = np.mean((temp_targets_np - temp_pred_np) ** 2, axis=1)  # MSE per node

    plt.hist(temp_errors, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=np.mean(temp_errors), color='r', linestyle='--',
                label=f'Mean MSE: {np.mean(temp_errors):.5f}')
    plt.title('Temperature Prediction Error Distribution')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Number of Nodes')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Visualize the graph with temperature at final timestep
    plt.subplot(2, 2, 4)

    # Convert to networkx graph
    G = to_networkx(graph_data, to_undirected=True)

    # Get node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)

    # Final temperatures (predicted and target)
    final_pred_temp = temp_pred_np[:, -1]
    final_target_temp = temp_targets_np[:, -1]

    # Normalize temperatures for color mapping
    vmin = min(np.min(final_pred_temp), np.min(final_target_temp))
    vmax = max(np.max(final_pred_temp), np.max(final_target_temp))

    # Draw predicted temperatures
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=final_pred_temp,
                          cmap=plt.cm.plasma, vmin=vmin, vmax=vmax,
                          label='Predicted')

    # Draw target temperatures as node borders
    node_border_colors = [final_target_temp[i] for i in range(num_nodes)]
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='none',
                          edgecolors=plt.cm.plasma(
                              (np.array(node_border_colors) - vmin) / (vmax - vmin + 1e-8)
                          ),
                          linewidths=2, alpha=0.7, label='Target')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title('Final Temperature Distribution\n(Fill: Predicted, Border: Target)')
    plt.axis('off')

    # Add a colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='Temperature')

    # Add overall metrics
    plt.figtext(0.5, 0.01,
                f"Overall Temperature MSE: {temp_mse:.6f} | Control MSE: {control_mse:.6f}",
                ha="center", fontsize=12,
                bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig('heat_diffusion_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Additional visualization: Temperature evolution over time
    plt.figure(figsize=(16, 8))

    # Select timesteps to visualize
    timesteps = [0, num_timesteps//4, num_timesteps//2, num_timesteps-1]

    for i, t in enumerate(timesteps):
        plt.subplot(2, len(timesteps), i+1)

        # Target temperature at timestep t
        nx.draw_networkx_nodes(G, pos, node_size=300,
                              node_color=temp_targets_np[:, t],
                              cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        plt.title(f'Target Temperature at t={t}')
        plt.axis('off')

        # Add a colorbar
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, cax=cax)

        plt.subplot(2, len(timesteps), i+len(timesteps)+1)

        # Predicted temperature at timestep t
        nx.draw_networkx_nodes(G, pos, node_size=300,
                              node_color=temp_pred_np[:, t],
                              cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        plt.title(f'Predicted Temperature at t={t}')
        plt.axis('off')

        # Add a colorbar
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.savefig('heat_diffusion_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

    return temp_mse, control_mse
