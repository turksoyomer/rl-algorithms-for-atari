import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class CategoricalNeuralNetwork(nn.Module):
    def __init__(self, algorithm_name, env_name, network_type, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor):
        """Initialization."""
        super(CategoricalNeuralNetwork, self).__init__()

        self.network_type = network_type
        self.parameter_path = f'./parameters/{algorithm_name}/{env_name}'
        if not os.path.exists(self.parameter_path):
            os.makedirs(self.parameter_path)

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

    def save_parameters(self, frame_idx):
        filename = self.parameter_path + f'/{self.network_type}_{frame_idx}.pth'
        torch.save(self.state_dict(), filename)

    def load_parameters(self, frame_idx):
        filename = self.parameter_path + f'/{self.network_type}_{frame_idx}.pth'
        parameters = torch.load(filename)
        self.load_state_dict(parameters)
