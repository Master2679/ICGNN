import sys,os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..models import HeatDiffusionICGRNN
# from icgrnn.models import HeatDiffusionICGRNN
from ..helper_scripts import train_heat_diffusion_model
from ..data import create_heat_diffusion_dataset
from ..visualization import visualize_model_performance

print("Generating synthetic heat diffusion data...")

graph_data, features, targets = create_heat_diffusion_dataset(
    num_nodes=50, num_timesteps=20, num_samples=500, diffusion_rate=0.1
)

# print(f"Generated dataset with {features.size(0)}samples")
# print(f"Graph has {graph_data.edge_index.max().item() + 1} nodes and {graph_data.edge_index.size(1)//2} edges")
# print(f"Feature shape: {features.shape}, Target shape: {targets.shape}")

model = HeatDiffusionICGRNN(
    input_dim=features.size(-1),  # Input dimension from dataset
    hidden_dim=32,                # Hidden dimension size
    output_dim=1,                 # Output dimension (typically 1 for temperature/control)
    icnn_hidden_dims=[32, 32,]     # Hidden dimensions in ICNNs
)


# Train the model
print("\nTraining the model...")
model = train_heat_diffusion_model(
    model, graph_data, features, targets, num_epochs=50, lr=0.001
)


temp_mse, control_mse = visualize_model_performance(model, graph_data, features, targets, sample_idx=0)

print("Control MSE:",temp_mse,"")
print("Control MSE:",control_mse,"")
