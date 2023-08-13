import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), -1)
        return x


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, algorithm_name, env_name, in_dim: int, out_dim: int):
        super(ActorCritic, self).__init__()

        self.parameter_path = f'./parameters/{algorithm_name}/{env_name}'
        if not os.path.exists(self.parameter_path):
            os.makedirs(self.parameter_path)

        self.actor = Actor(in_dim, out_dim)
        self.critic = Critic(in_dim)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def save_parameters(self, time_step):
        filename = self.parameter_path + f'/actor_critic_{time_step}.pth'
        torch.save(self.state_dict(), filename)

    def load_parameters(self, time_step):
        filename = self.parameter_path + f'/actor_critic_{time_step}.pth'
        parameters = torch.load(filename, map_location=lambda storage, loc: storage)
        self.load_state_dict(parameters)
    