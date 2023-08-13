import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_dim: int):
        super(ConvolutionalLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class DenseLayer(nn.Module):
    def __init__(self, out_dim: int):
        super(DenseLayer, self).__init__()

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, algorithm_name, env_name, network_type, in_dim: int, out_dim: int):
        """Initialization."""
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.network_type = network_type
        self.parameter_path = f'./parameters/{algorithm_name}/{env_name}'
        if not os.path.exists(self.parameter_path):
            os.makedirs(self.parameter_path)
        
        self.conv = ConvolutionalLayer(in_dim)
        self.dense = DenseLayer(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.conv(x)
        x = self.dense(x)
        return x

    def save_parameters(self, frame_idx):
        filename = self.parameter_path + f'/{self.network_type}_{frame_idx}.pth'
        torch.save(self.state_dict(), filename)

    def load_parameters(self, frame_idx):
        filename = self.parameter_path + f'/{self.network_type}_{frame_idx}.pth'
        parameters = torch.load(filename)
        self.load_state_dict(parameters)
