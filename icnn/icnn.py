# icnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICNN(nn.Module):
    """
    Input-Convex Neural Network:
      • primary linear layers (no bias) guaranteed non-negative via softplus
      • skip connections from the raw input (with bias allowed)
      • ReLU activations to preserve convexity
    """
    def __init__(self,
                 input_dim: int = 28*28,
                 hidden_dims: list[int] = [512, 512, 512, 512],
                 output_dim: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_dim = input_dim

        # build primary (no-bias) + skip layers
        dims = [input_dim] + hidden_dims
        self.primary_layers = nn.ModuleList()
        self.skip_layers    = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            # primary path (no bias)
            prim = nn.Linear(in_d, out_d, bias=False)
            # skip path from the original input (bias allowed)
            skip = nn.Linear(input_dim, out_d, bias=True)
            # initialize weights positive
            nn.init.kaiming_uniform_(prim.weight, nonlinearity='relu')
            prim.weight.data.abs_()
            self.primary_layers.append(prim)
            self.skip_layers.append(skip)

        # final output layer (no bias) + skip
        self.output_prim = nn.Linear(hidden_dims[-1], output_dim, bias=False)
        self.output_skip = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.kaiming_uniform_(self.output_prim.weight, nonlinearity='relu')
        self.output_prim.weight.data.abs_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten once
        device = self.primary_layers[0].weight.device
        x0 = self.flatten(x).to(device)  # [batch, input_dim]

        # first hidden
        z = F.relu(
            F.linear(x0, F.softplus(self.primary_layers[0].weight), None)
            + self.skip_layers[0](x0)
        )

        # subsequent hidden layers
        for prim, skip in zip(self.primary_layers[1:], self.skip_layers[1:]):
            z = F.relu(
                F.linear(z, F.softplus(prim.weight), None)
                + skip(x0)
            )

        # final output (convex)
        out = F.linear(z, F.softplus(self.output_prim.weight), None)
        out = out + self.output_skip(x0)
        return out