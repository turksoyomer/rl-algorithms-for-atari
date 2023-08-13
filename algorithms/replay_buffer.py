from typing import Dict
import numpy as np
import pickle
import os

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, algorithm_name: str, env_name: str, memory_name: str, obs_dim: tuple, size: int, batch_size: int = 32):
        self.memory_name = memory_name
        self.size = size,
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.memory_path = f'./memories/{algorithm_name}/{env_name}'
        if not os.path.exists(self.memory_path):
            os.makedirs(self.memory_path)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def save_memory(self):
        memory = {
            'states': self.obs_buf,
            'next_states': self.next_obs_buf,
            'actions': self.acts_buf,
            'rewards': self.rews_buf,
            'dones': self.done_buf,
            'ptr': self.ptr,
            'size': self.size,
        }
        filename = self.memory_path + f'/{self.memory_name}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(memory, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_memory(self):
        filename = self.memory_path + f'/{self.memory_name}.pkl'
        with open(filename, 'rb') as file:
            memory = pickle.load(file)
            self.obs_buf = memory.get('states', self.obs_buf)
            self.next_obs_buf = memory.get('next_states', self.next_obs_buf)
            self.acts_buf = memory.get('actions', self.acts_buf)
            self.rews_buf = memory.get('rewards', self.rews_buf)
            self.done_buf = memory.get('dones', self.done_buf)
            self.ptr = memory.get('ptr', self.ptr)
            self.size = memory.get('size', self.size)

    def __len__(self) -> int:
        return self.size