import random
from typing import Dict, List
import numpy as np
from utils.segment_tree import MinSegmentTree, SumSegmentTree
from .replay_buffer import ReplayBuffer
import pickle

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        algorithm_name: str,
        env_name: str,
        memory_name: str,
        obs_dim: tuple,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6
    ):
        """Initialization."""
        self.memory_name = memory_name
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(algorithm_name, env_name, memory_name, obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        self.tree_capacity = 1
        while self.tree_capacity < self.max_size:
            self.tree_capacity *= 2

        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

    def save_memory(self):
        memory = {
            'states': self.obs_buf,
            'next_states': self.next_obs_buf,
            'actions': self.acts_buf,
            'rewards': self.rews_buf,
            'dones': self.done_buf,
            'ptr': self.ptr,
            'size': self.size,
            'max_priority': self.max_priority,
            'tree_ptr': self.tree_ptr,
            'alpha': self.alpha,
            'tree_capacity': self.tree_capacity,
            'sum_tree': self.sum_tree,
            'min_tree': self.min_tree,
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
            self.max_priority = memory.get('max_priority', self.max_priority)
            self.tree_ptr = memory.get('tree_ptr', self.tree_ptr)
            self.alpha = memory.get('alpha', self.alpha)
            self.tree_capacity = memory.get('tree_capacity', self.tree_capacity)
            self.sum_tree = memory.get('sum_tree', self.sum_tree)
            self.min_tree = memory.get('min_tree', self.min_tree)
