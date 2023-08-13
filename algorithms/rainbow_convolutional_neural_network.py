import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

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


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class RainbowConvolutionalNeuralNetwork(nn.Module):
    def __init__(
        self, 
        algorithm_name: str, 
        env_name: str, 
        network_type: str, 
        in_dim: int, 
        out_dim: int,
        atom_size: int,
        support: torch.Tensor,
    ):
        """Initialization."""
        super(RainbowConvolutionalNeuralNetwork, self).__init__()

        self.network_type = network_type
        self.parameter_path = f'./parameters/{algorithm_name}/{env_name}'
        if not os.path.exists(self.parameter_path):
            os.makedirs(self.parameter_path)
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.conv = ConvolutionalLayer(in_dim)
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(64 * 7 * 7, 512)
        self.advantage_layer = NoisyLinear(512, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(64 * 7 * 7, 512)
        self.value_layer = NoisyLinear(512, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        x = self.conv(x)
        x = x.view(-1, 64 * 7 * 7)
        adv_hid = F.relu(self.advantage_hidden_layer(x))
        val_hid = F.relu(self.value_hidden_layer(x))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

    def save_parameters(self, frame_idx):
        filename = self.parameter_path + f'/{self.network_type}_{frame_idx}.pth'
        torch.save(self.state_dict(), filename)

    def load_parameters(self, frame_idx):
        filename = self.parameter_path + f'/{self.network_type}_{frame_idx}.pth'
        parameters = torch.load(filename)
        self.load_state_dict(parameters)
