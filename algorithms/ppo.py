import os
import time
import glob
import pickle
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from .rollout_buffer import RolloutBuffer
from PIL import Image

class PPOAgent:
    def __init__(
        self, 
        env: gym.Env, 
        K_epochs: int, 
        lr_actor: float = 0.0003, 
        lr_critic: float = 0.001, 
        gamma: float = 0.99, 
        eps_clip: float = 0.2,
        use_conv: bool = True,
    ):
        self.algorithm_name = 'ppo'
        self.env = env
        self.env_name = env.spec.id.split('/')[-1]

        self.obs_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma # discount factor
        self.eps_clip = eps_clip # clip parameter for PPO
        self.K_epochs = K_epochs # update policy for K epochs in one PPO update
        self.use_conv = use_conv

        self.time_step = 0
        self.i_episode = 0
        
        self.buffer = RolloutBuffer()

        self.variable_path = f'./variables/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.variable_path):
            os.makedirs(self.variable_path)
        self.plot_path = f'./plots/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        self.optimizer_path = f'./parameters/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.optimizer_path):
            os.makedirs(self.optimizer_path)
        self.log_path = f'./logs/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.gif_images_path = f'./gif_images/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.gif_images_path):
            os.makedirs(self.gif_images_path)
        self.gif_path = f'./gifs/{self.algorithm_name}/{self.env_name}'
        if not os.path.exists(self.gif_path):
            os.makedirs(self.gif_path)

        ################################## set device ##################################
        print("===================================================================================")
        # set device to cpu or cuda
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("===================================================================================")

        if use_conv:
            from .convolutional_actor_critic import ActorCritic
        else:
            from .actor_critic import ActorCritic

        self.policy = ActorCritic(self.algorithm_name, self.env_name, self.obs_dim[0], 
                                  self.action_dim).to(self.device)
        self.policy_old = ActorCritic(self.algorithm_name, self.env_name, self.obs_dim[0], 
                                      self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.MseLoss = nn.MSELoss()

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        if not self.is_test:
            self.transition = [state, action, action_logprob, state_val]

        return action.item()
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        next_state = np.array(next_state)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, done]
            self.buffer.store(*self.transition)

        return next_state, reward, done
    
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), 
                                       reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach() \
            .to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach() \
            .to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach() \
            .to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + \
                0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save_agent(self):
        self.policy_old.save_parameters(self.time_step)
        optimizer_filename = self.optimizer_path + f'/optimizer_{self.time_step}.pth'
        torch.save(self.optimizer.state_dict(), optimizer_filename)
        variables = {
            'time_step': self.time_step,
            'i_episode': self.i_episode,
        }
        variable_filename = self.variable_path + f'/variables_{self.time_step}.pkl'
        with open(variable_filename, 'wb') as file:
            pickle.dump(variables, file, protocol=pickle.HIGHEST_PROTOCOL)
   
    def load_agent(self, time_step: int):
        self.policy_old.load_parameters(time_step)
        self.policy.load_parameters(time_step)
        optimizer_filename = self.optimizer_path + f'/optimizer_{time_step}.pth'
        optimizer_parameters = torch.load(optimizer_filename)
        self.optimizer.load_state_dict(optimizer_parameters)
        variable_filename = self.variable_path + f'/variables_{time_step}.pkl'
        with open(variable_filename, 'rb') as file:
            variables = pickle.load(file)
            self.time_step = variables.get('time_step', self.time_step)
            self.i_episode = variables.get('i_episode', self.i_episode)
        print('Agent loaded successfuly.')

    def train(self, max_ep_len=1000, max_training_timesteps=int(3e6), save_model_freq=int(1e5)):
        """
        max_ep_len: max timesteps in one episode
        max_training_timesteps: break training loop if timeteps > max_training_timesteps
        save_model_freq: save model frequency (in num timesteps)
        """
        print("===================================================================================")

        print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
        log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)

        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        update_timestep = max_ep_len * 4      # update policy every n timesteps

        random_seed = 0         # set random seed if required (0 = no random seed)
        #####################################################

        print("training environment name : " + self.env_name)

        ###################### logging ######################

        #### get number of log files in log directory
        current_num_files = next(os.walk(self.log_path))[2]
        if self.time_step != 0:
            run_num = len(current_num_files) - 1
        else:
            run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = self.log_path + f'/log_{run_num}.csv'

        print("current logging run number for " + self.env_name + " : ", run_num)
        print("logging at : " + log_f_name)
        #####################################################

        ################### checkpointing ###################

        checkpoint_path = self.policy_old.parameter_path
        print("save checkpoint path : " + checkpoint_path)
        #####################################################


        ############# print all hyperparameters #############
        print("-----------------------------------------------------------------------------------")
        print("max training timesteps : ", max_training_timesteps)
        print("max timesteps per episode : ", max_ep_len)
        print("model saving frequency : " + str(save_model_freq) + " timesteps")
        print("log frequency : " + str(log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
        print("-----------------------------------------------------------------------------------")
        print("state space dimension : ", self.obs_dim)
        print("action space dimension : ", self.action_dim)
        print("-----------------------------------------------------------------------------------")
        print("Initializing a discrete action space policy")
        print("-----------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(update_timestep) + " timesteps")
        print("PPO K epochs : ", self.K_epochs)
        print("PPO epsilon clip : ", self.eps_clip)
        print("discount factor (gamma) : ", self.gamma)
        print("-----------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", self.lr_actor)
        print("optimizer learning rate critic : ", self.lr_critic)
        if random_seed:
            print("-------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            self.env.seed(random_seed)
            np.random.seed(random_seed)
        #####################################################

        print("===================================================================================")

        ################# training procedure ################

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("===================================================================================")

        # logging file
        log_f = open(log_f_name,"a+")
        if os.stat(log_f_name).st_size == 0:
            log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        # training loop
        while self.time_step <= max_training_timesteps:

            state, _ = self.env.reset()
            state = np.array(state)
            current_ep_reward = 0

            for t in range(1, max_ep_len+1):

                # select action with policy
                action = self.select_action(state)
                state, reward, done = self.step(action)

                self.time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if self.time_step % update_timestep == 0:
                    self.update()

                # log in logging file
                if self.time_step % log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(self.i_episode, self.time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if self.time_step % print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}" \
                          .format(self.i_episode, self.time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if self.time_step % save_model_freq == 0:
                    print("-----------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    self.save_agent()
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("-----------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            self.i_episode += 1

        log_f.close()
        self.env.close()

        # print total training time
        print("===================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("===================================================================================")

    def test(self, max_ep_len=1000):
        """
        max_ep_len: max timesteps in one episode
        """
        print("===================================================================================")

        ################## hyperparameters ##################

        render = True              # render environment on screen
        frame_delay = 0             # if required; add delay b/w frames

        total_test_episodes = 10    # total num of testing episodes

        #####################################################

        # preTrained weights directory

        print("-----------------------------------------------------------------------------------")

        test_running_reward = 0

        for ep in range(1, total_test_episodes+1):
            ep_reward = 0
            state, _ = self.env.reset()
            state = np.array(state)

            for t in range(1, max_ep_len+1):
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                state = np.array(state)
                done = terminated or truncated
                ep_reward += reward

                if render:
                    self.env.render()
                    time.sleep(frame_delay)

                if done:
                    break

            # clear buffer
            self.buffer.clear()

            test_running_reward +=  ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            ep_reward = 0

        self.env.close()

        print("===================================================================================")

        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)
        print("average test reward : " + str(avg_test_reward))

        print("===================================================================================")

    def save_gif_images(self, max_ep_len=1000):
        print("===================================================================================")

        total_test_episodes = 1     # save gif for only one episode

        print("-----------------------------------------------------------------------------------")
        test_running_reward = 0

        for ep in range(1, total_test_episodes+1):

            ep_reward = 0
            state, _ = self.env.reset()
            state = np.array(state)

            for t in range(1, max_ep_len+1):
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                state = np.array(state)
                done = terminated or truncated
                ep_reward += reward

                img = self.env.render()

                img = Image.fromarray(img)
                img.save(self.gif_images_path + '/' + str(t).zfill(6) + '.jpg')

                if done:
                    break

            # clear buffer
            self.buffer.clear()

            test_running_reward +=  ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            ep_reward = 0

        self.env.close()

        print("===================================================================================")
        print("total number of frames / timesteps / images saved : ", t)
        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)
        print("average test reward : " + str(avg_test_reward))
        print("===================================================================================")

    def save_gif(self):
        ######################## generate gif from saved images ########################

        print("===================================================================================")

        gif_num = 0     #### change this to prevent overwriting gifs in same env_name folder

        # adjust following parameters to get desired duration, size (bytes) and smoothness of gif
        total_timesteps = 300
        step = 1
        frame_duration = 15

        # input images
        gif_images_dir = self.gif_images_path + '/*.jpg'

        gif_path = self.gif_path + '/gif_' + str(gif_num) + '.gif'

        img_paths = sorted(glob.glob(gif_images_dir))
        img_paths = img_paths[:total_timesteps]
        img_paths = img_paths[::step]

        print("total frames in gif : ", len(img_paths))
        print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) \
              + " seconds")

        # save gif
        img, *imgs = [Image.open(f) for f in img_paths]
        img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, 
                 duration=frame_duration, loop=0)

        print("saved gif at : ", gif_path)

        print("===================================================================================")

    def list_gif_size(self):
        ############################# check gif byte size ##############################
        print("===================================================================================")
        gif_dir = self.gif_path + '/*.gif'
        gif_paths = sorted(glob.glob(gif_dir))
        for gif_path in gif_paths:
            file_size = os.path.getsize(gif_path)
            print(gif_path + '\t\t' + str(round(file_size / (1024 * 1024), 2)) + " MB")
        print("===================================================================================")

    def save_graph(self):
        print("===================================================================================")

        fig_num = 0     #### change this to prevent overwriting figures in same env_name folder
        plot_avg = True    # plot average of all runs; else plot all runs separately
        fig_width = 10
        fig_height = 6

        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        window_len_smooth = 20
        min_window_len_smooth = 1
        linewidth_smooth = 1.5
        alpha_smooth = 1

        window_len_var = 5
        min_window_len_var = 1
        linewidth_var = 2
        alpha_var = 0.1

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 
                  'crimson','gray', 'black']

        fig_save_path = self.plot_path + '/fig_' + str(fig_num) + '.png'


        # get number of log files in directory
        current_num_files = next(os.walk(self.log_path + '/'))[2]
        num_runs = len(current_num_files)

        all_runs = []

        for run_num in range(num_runs):

            log_f_name = self.log_path + '/log_' + str(run_num) + ".csv"
            print("loading data from : " + log_f_name)
            data = pd.read_csv(log_f_name)
            data = pd.DataFrame(data)

            print("data shape : ", data.shape)

            all_runs.append(data)
            print("-------------------------------------------------------------------------------")

        ax = plt.gca()

        if plot_avg:
            # average all runs
            df_concat = pd.concat(all_runs)
            df_concat_groupby = df_concat.groupby(df_concat.index)
            data_avg = df_concat_groupby.mean()

            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            data_avg['reward_smooth'] = data_avg['reward'] \
                .rolling(window=window_len_smooth, win_type='triang', 
                         min_periods=min_window_len_smooth).mean()
            data_avg['reward_var'] = data_avg['reward'] \
                .rolling(window=window_len_var, win_type='triang', 
                         min_periods=min_window_len_var).mean()

            data_avg.plot(kind='line', x='timestep' , y='reward_smooth',ax=ax,color=colors[0],  
                          linewidth=linewidth_smooth, alpha=alpha_smooth)
            data_avg.plot(kind='line', x='timestep' , y='reward_var',ax=ax,color=colors[0],  
                          linewidth=linewidth_var, alpha=alpha_var)

            # keep only reward_smooth in the legend and rename it
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2)

        else:
            for i, run in enumerate(all_runs):
                # smooth out rewards to get a smooth and a less smooth (var) plot lines
                run['reward_smooth_' + str(i)] = run['reward'] \
                    .rolling(window=window_len_smooth, win_type='triang', 
                             min_periods=min_window_len_smooth).mean()
                run['reward_var_' + str(i)] = run['reward'] \
                    .rolling(window=window_len_var, win_type='triang', 
                             min_periods=min_window_len_var).mean()

                # plot the lines
                run.plot(kind='line', x='timestep' , 
                         y='reward_smooth_' + str(i),ax=ax,color=colors[i % len(colors)], 
                         linewidth=linewidth_smooth, alpha=alpha_smooth)
                run.plot(kind='line', x='timestep', 
                         y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)], 
                         linewidth=linewidth_var, alpha=alpha_var)

            # keep alternate elements (reward_smooth_i) in the legend
            handles, labels = ax.get_legend_handles_labels()
            new_handles = []
            new_labels = []
            for i in range(len(handles)):
                if(i%2 == 0):
                    new_handles.append(handles[i])
                    new_labels.append(labels[i])
            ax.legend(new_handles, new_labels, loc=2)

        # ax.set_yticks(np.arange(0, 1800, 200))
        # ax.set_xticks(np.arange(0, int(4e6), int(5e5)))

        ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Rewards", fontsize=12)

        plt.title(self.env_name, fontsize=14)

        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_height)

        print("===================================================================================")
        plt.savefig(fig_save_path)
        print("figure saved at : ", fig_save_path)
        print("===================================================================================")
        
        plt.show()
        