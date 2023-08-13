import maps
import gymnasium as gym

class Trainer:
    def __init__(self, env_id, algorithm, num_frames: int, last_trained_frame: int = None, 
                 saving_interval=10000, is_test=False):
        assert env_id in maps.ENVIRONMENTS
        assert algorithm in maps.ALGORITHMS[env_id]
        
        self.num_frames = num_frames
        self.saving_interval = saving_interval
        self.is_test = is_test

        self.algorithm = algorithm

        if env_id.startswith("ALE"):
            self.env = gym.make(env_id, obs_type="grayscale", 
                                render_mode="rgb_array" if is_test else None)
            self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
            self.env = gym.wrappers.FrameStack(self.env, num_stack=4)
            self.use_conv = True
        else:
            self.env = gym.make(env_id, render_mode="rgb_array" if is_test else None)
            self.use_conv = False
        
        if algorithm == "DQN":
            from algorithms.dqn import DQNAgent
            memory_size = 1000
            batch_size = 32
            target_update = 100
            epsilon_decay = 1 / 2000
            self.agent = DQNAgent(self.env, memory_size, batch_size, target_update, epsilon_decay, 
                             use_conv=self.use_conv)
        elif algorithm == "Double DQN":
            from algorithms.double_dqn import DoubleDQNAgent
            memory_size = 1000
            batch_size = 32
            target_update = 200
            epsilon_decay = 1 / 2000
            self.agent = DoubleDQNAgent(self.env, memory_size, batch_size, target_update, 
                                        epsilon_decay, use_conv=self.use_conv)
        elif algorithm == "PER DQN":
            from algorithms.per_dqn import PERDQNAgent
            memory_size = 2000
            batch_size = 32
            target_update = 100
            epsilon_decay = 1 / 2000
            self.agent = PERDQNAgent(self.env, memory_size, batch_size, target_update, 
                                     epsilon_decay, use_conv=self.use_conv)
        elif algorithm == "Dueling DQN":
            from algorithms.dueling_dqn import DuelingDQNAgent
            memory_size = 1000
            batch_size = 32
            target_update = 100
            epsilon_decay = 1 / 2000
            self.agent = DuelingDQNAgent(self.env, memory_size, batch_size, target_update, 
                                         epsilon_decay, use_conv=self.use_conv)
        elif algorithm == "Noisy DQN":
            from algorithms.noisy_dqn import NoisyDQNAgent
            memory_size = 10000
            batch_size = 128
            target_update = 150
            self.agent = NoisyDQNAgent(self.env, memory_size, batch_size, target_update, 
                                  use_conv=self.use_conv)
        elif algorithm == "Categorical DQN":
            from algorithms.categorical_dqn import CategoricalDQNAgent
            memory_size = 2000
            batch_size = 32
            target_update = 200
            epsilon_decay = 1 / 2000
            self.agent = CategoricalDQNAgent(self.env, memory_size, batch_size, target_update, 
                                        epsilon_decay, use_conv=self.use_conv)
        elif algorithm == "N-Step DQN":
            from algorithms.nstep_dqn import NStepDQNAgent
            memory_size = 2000
            batch_size = 32
            target_update = 100
            epsilon_decay = 1 / 2000
            self.agent = NStepDQNAgent(self.env, memory_size, batch_size, target_update, 
                                       epsilon_decay, use_conv=self.use_conv)
        elif algorithm == "Rainbow":
            from algorithms.rainbow import RainbowAgent
            memory_size = 10000
            batch_size = 128
            target_update = 100
            self.agent = RainbowAgent(self.env, memory_size, batch_size, target_update, 
                                 use_conv=self.use_conv)
        elif algorithm == "PPO":
            from algorithms.ppo import PPOAgent
            K_epochs = 80
            self.agent = PPOAgent(self.env, K_epochs, use_conv=self.use_conv)

        if last_trained_frame:
            self.agent.load_agent(last_trained_frame)

    def train_agent(self):
        if self.algorithm == "PPO":
            self.agent.train(max_training_timesteps=self.num_frames, 
                             save_model_freq=self.saving_interval)
        else:
            self.agent.train(self.num_frames, self.saving_interval)

    def save(self):
        self.agent.save_agent()
    
    def plot(self, frame_idx: int):
        if self.algorithm == "PPO":
            self.agent.save_graph()
        else:
            self.agent._plot(frame_idx)
    
    def test_agent(self):
        if not self.test_agent:
            print('Trainer should be in the test mode.')
            return
        self.agent.test()

    def make_gif(self):
        if not self.test_agent:
            print('Trainer should be in the test mode.')
            return
        self.agent.save_gif_images()
        self.agent.save_gif()
        self.agent.list_gif_size()
        