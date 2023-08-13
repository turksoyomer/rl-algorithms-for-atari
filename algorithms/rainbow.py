from typing import Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from .nstep_replay_buffer import NStepReplayBuffer
from .nstep_prioritized_replay_buffer import NStepPrioritizedReplayBuffer
from .rainbow_neural_network import RainbowNeuralNetwork
from .rainbow_convolutional_neural_network import RainbowConvolutionalNeuralNetwork
import pickle
import os
from torch.nn.utils import clip_grad_norm_

class RainbowAgent:
    """Rainbow Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (NStepPrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        behavior_network (Network): model to train and select actions
        target_network (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (NStepReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        use_conv: bool = True,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
            use_conv (bool): use conv layer or not
        """
        obs_dim = env.observation_space.shape
        action_dim = env.action_space.n
        
        self.algorithm_name = 'rainbow'
        self.env = env
        self.env_name = env.spec.id.split('/')[-1]
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        self.use_conv = use_conv
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = NStepPrioritizedReplayBuffer(
            self.algorithm_name, self.env_name, 'memory', obs_dim, memory_size, batch_size, alpha=alpha
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = NStepReplayBuffer(
                self.algorithm_name, self.env_name, 'memory_n', obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        self.update_cnt = 0
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
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: behavior_network, target_network
        if use_conv:
            self.behavior_network = RainbowConvolutionalNeuralNetwork(self.algorithm_name, self.env_name, 'behavior', obs_dim[0], action_dim, self.atom_size, self.support).to(self.device)
            self.target_network = RainbowConvolutionalNeuralNetwork(self.algorithm_name, self.env_name, 'target', obs_dim[0], action_dim, self.atom_size, self.support).to(self.device)
        else:
            self.behavior_network = RainbowNeuralNetwork(self.algorithm_name, self.env_name, 'behavior', obs_dim[0], action_dim, self.atom_size, self.support).to(self.device)
            self.target_network = RainbowNeuralNetwork(self.algorithm_name, self.env_name, 'target', obs_dim[0], action_dim, self.atom_size, self.support).to(self.device)
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
        # NoisyNet: no epsilon greedy action selection
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
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.behavior_network.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.behavior_network.reset_noise()
        self.target_network.reset_noise()

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
            
            # NoisyNet: removed decrease of epsilon
            
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
        self.memory_n.save_memory()
        self.behavior_network.save_parameters(self.start_frame)
        self.target_network.save_parameters(self.start_frame)
        optimizer_filename = self.optimizer_path + f'/optimizer_{self.start_frame}.pth'
        torch.save(self.optimizer.state_dict(), optimizer_filename)
        variables = {
            'update_cnt': self.update_cnt,
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
        self.memory_n.load_memory()
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
            self.update_cnt = variables.get('update_cnt', self.update_cnt)
            self.losses = variables.get('losses', self.losses)
            self.scores = variables.get('scores', self.scores)
            self.start_frame = variables.get('start_frame', 0) + 1
            self.beta = variables.get('beta', self.beta)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.behavior_network(next_state).argmax(1)
            next_dist = self.target_network.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.behavior_network.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

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
        filename = self.plot_path + f'/plot_{frame_idx}.png'
        plt.savefig(filename)
        plt.show()
