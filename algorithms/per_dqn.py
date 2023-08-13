from typing import Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .neural_network import NeuralNetwork
from .convolutional_neural_network import ConvolutionalNeuralNetwork
import pickle
import os

class PERDQNAgent:
    """DQN Agent with prioritized replay buffer interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        behavior_network (Network): model to train and select actions
        target_network (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        beta (float): determines how much importance sampling is used
        prior_eps (float): guarantees every transition can be sampled
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        use_conv: bool = True,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
            use_conv (bool): use conv layer or not
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
        """
        obs_dim = env.observation_space.shape
        action_dim = env.action_space.n
        
        self.algorithm_name = 'per_dqn'
        self.env = env
        self.env_name = env.spec.id.split('/')[-1]
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.use_conv = use_conv

        self.update_cnt = 0
        self.epsilons = []
        self.losses = []
        self.scores = []
        self.start_frame = 1
        self.variable_path = f'./variables/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.variable_path):
            os.makedirs(self.variable_path)
        self.plot_path = f'./plots/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        self.test_video_path = f'./videos/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.test_video_path):
            os.makedirs(self.test_video_path)
        self.optimizer_path = f'./parameters/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.optimizer_path):
            os.makedirs(self.optimizer_path)
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            self.algorithm_name, self.env_name, 'memory', obs_dim, memory_size, batch_size, alpha
        )

        # networks: behavior_network, target_network
        if use_conv:
            self.behavior_network = ConvolutionalNeuralNetwork(self.algorithm_name, self.env_name, 'behavior', obs_dim[0], action_dim).to(self.device)
            self.target_network = ConvolutionalNeuralNetwork(self.algorithm_name, self.env_name, 'target', obs_dim[0], action_dim).to(self.device)
        else:
            self.behavior_network = NeuralNetwork(self.algorithm_name, self.env_name, 'behavior', obs_dim[0], action_dim).to(self.device)
            self.target_network = NeuralNetwork(self.algorithm_name, self.env_name, 'target', obs_dim[0], action_dim).to(self.device)
        self.target_network.load_state_dict(self.behavior_network.state_dict())
        self.target_network.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.behavior_network.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.behavior_network(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_observation, reward, terminated, truncated, _ = self.env.step(action)
        next_state = np.array(next_observation)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()
        
    def train(self, num_frames: int, saving_interval: int = 10000):
        """Train the agent."""
        self.is_test = False
        
        observation, _ = self.env.reset()
        state = np.array(observation)
        score = 0

        for frame_idx in range(self.start_frame, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                observation, _ = self.env.reset()
                state = np.array(observation)
                self.scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                self.losses.append(loss)
                self.update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                self.epsilons.append(self.epsilon)
                
                # if hard update is needed
                if self.update_cnt % self.target_update == 0:
                    self._target_hard_update()

            self.start_frame = frame_idx

            # saving
            if frame_idx % saving_interval == 0:
                self.save_agent()
                
        self.env.close()
                
    def test(self) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=self.test_video_path)
        
        observation, _ = self.env.reset()
        state = np.array(observation)
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env

    def save_agent(self):
        self.memory.save_memory()
        self.behavior_network.save_parameters(self.start_frame)
        self.target_network.save_parameters(self.start_frame)
        optimizer_filename = self.optimizer_path + f'/optimizer_{self.start_frame}.pth'
        torch.save(self.optimizer.state_dict(), optimizer_filename)
        variables = {
            'epsilon': self.epsilon,
            'update_cnt': self.update_cnt,
            'epsilons': self.epsilons,
            'losses': self.losses,
            'scores': self.scores,
            'start_frame': self.start_frame,
            'beta': self.beta,
        }
        filename = self.variable_path + f'/variables_{self.start_frame}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(variables, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_agent(self, frame_idx):
        self.memory.load_memory()
        self.behavior_network.load_parameters(frame_idx)
        self.behavior_network.to(self.device)
        self.target_network.load_parameters(frame_idx)
        self.target_network.to(self.device)
        self.target_network.eval()
        optimizer_filename = self.optimizer_path + f'/optimizer_{frame_idx}.pth'
        optimizer_parameters = torch.load(optimizer_filename)
        self.optimizer.load_state_dict(optimizer_parameters)
        filename = self.variable_path + f'/variables_{frame_idx}.pkl'
        with open(filename, 'rb') as file:
            variables = pickle.load(file)
            self.epsilon = variables.get('epsilon', self.epsilon)
            self.update_cnt = variables.get('update_cnt', self.update_cnt)
            self.epsilons = variables.get('epsilons', self.epsilons)
            self.losses = variables.get('losses', self.losses)
            self.scores = variables.get('scores', self.scores)
            self.start_frame = variables.get('start_frame', 0) + 1
            self.beta = variables.get('beta', self.beta)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.behavior_network(state).gather(1, action)
        next_q_value = self.target_network(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target_network.load_state_dict(self.behavior_network.state_dict())
                
    def _plot(
        self, 
        frame_idx: int,
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(self.scores[-10:])))
        plt.plot(self.scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(self.losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(self.epsilons)
        filename = self.plot_path + f'/plot_{frame_idx}.png'
        plt.savefig(filename)
        plt.show()
