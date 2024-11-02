#pip3 install gymnasium
#pip3 install "gymnasium[classic-control]"

# # to see a list of all available gyms use...
# import gymnasium as gym
# gym.pprint_registry()

# see also https://github.com/openai/gym/wiki/Leaderboard

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import shutil
import uuid
import matplotlib.gridspec as gridspec
import numpy as np
import json

import rl_utils as rl

if not os.getcwd().endswith("data"):
  os.chdir("data")
  print(F"Set current folder to {os.getcwd()}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, settings):
        super(DQN, self).__init__()
        
        linear_0 = settings[0]
        linear_1 = settings[1]
        linear_2 = settings[2]
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_observations, linear_0),
            nn.ReLU(),
            nn.Linear(linear_0, linear_1),
            nn.ReLU(),
            nn.Linear(linear_1, linear_2),
            nn.ReLU(),
            nn.Linear(linear_2, n_actions),
            )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class PT_GYM():
  def __init__(self, gym_name, render_mode=None):
    self.gym_name = gym_name
    self.env = gym.make(self.gym_name, render_mode=render_mode)
   
  def reset(self):
    return self.env.reset()
  def step(self, action):
    #print(f"step. action = {action}")
    return self.env.step(action)
  def render(self):
    return self.env.render()
  def close(self):
    return self.env.close()

class PT_DQN():
  def __init__(self, creat_env_fn, settings):
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    self.BATCH_SIZE = 128
    self.GAMMA = 0.99
    self.EPS_START = 0.9
    self.EPS_END = 0.05
    self.EPS_DECAY = 5000 #1000
    self.TAU = 0.005
    self.LR = 1e-4
    self.settings = settings
    
    self.meta_state = rl.MetaState([
                      "episodes",
                      "steps_done",
                      "epsilon",
                      "memory_size",
                      "episode_durations",
                      "reward_total",
                      ])
    
    self.creat_env_fn = creat_env_fn
    self.env = self.creat_env_fn()
    self.gym_name = self.env.gym_name
    self.action_space_is_discrete = ( True if f"{self.env.env.action_space}".startswith("Discrete") else False )

    # Get number of actions from gym action space
    if self.action_space_is_discrete:
      self.n_actions = self.env.env.action_space.n
    else:
      self.n_actions = self.env.env.action_space.shape[0]

    # Get the number of state observations
    state, info = self.env.reset()
    self.n_observations = len(state)

    # # set up matplotlib
    # is_ipython = 'inline' in matplotlib.get_backend()
    # if is_ipython:
        # from IPython import display
    # plt.ion()

    # if GPU is to be used
    self.device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device = {self.device}")

    self.policy_net = DQN(self.n_observations, self.n_actions, self.settings).to(self.device)
    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
    self.memory = ReplayMemory(10000)

    self.steps_done = 0
    self.average_duration = 0

    self.load_if_save_file_present()
    
    self.target_net = DQN(self.n_observations, self.n_actions, self.settings).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def get_save_name(self):
    filename = "rl_"+self.gym_name+"_"
    for setting in self.settings:
      filename += str(setting)+"_"
    return filename

  def save(self):
    temp_filename = "_"+str(uuid.uuid4())
    torch.save(self.policy_net.state_dict(), temp_filename+".tempnet")
    torch.save(self.memory, temp_filename+".tempmem")
    with open(temp_filename+".tempjson", 'w') as f:
      json.dump({
                'settings': self.settings,
                'meta_state': self.meta_state.to_dict(),
                }, f, indent=2)
    filename = self.get_save_name()
    if os.path.isfile(filename+".net"):
      os.remove(filename+".net")
    if os.path.isfile(filename+".mem"):
      os.remove(filename+".mem")
    if os.path.isfile(filename+".json"):
      os.remove(filename+".json")
    os.rename(temp_filename+".tempnet",filename+".net")
    os.rename(temp_filename+".tempmem",filename+".mem")
    os.rename(temp_filename+".tempjson",filename+".json")
    print(f"Saved {filename} ({len(self.meta_state)} episodes)")

  def load_if_save_file_present(self):
    filename = self.get_save_name()
    if os.path.isfile(filename+".net"):
      self.load(filename)
    else:
      print(f"File {filename} not found")

  def load(self,filename):
    print(f"Loading {filename} ...")
    load_net = torch.load(filename+".net",weights_only=False)
    self.policy_net.load_state_dict(load_net)
    self.policy_net.eval()  # Set the model to evaluation mode
    load_mem = torch.load(filename+".mem",weights_only=False)
    self.memory = load_mem
    with open(filename+".json", 'r') as f:
      load_json = json.load(f)
      self.meta_state = rl.MetaState.from_dict(load_json['meta_state'])
      self.settings = load_json['settings']

    # any fix-ups from loading the meta-state
    self.steps_done = self.meta_state.get_latest_value("steps_done")
    
    print(f"Loaded {filename} ({len(self.meta_state)} episodes)")

  def get_policy_action(self,state):
    with torch.no_grad():
      raw_action = self.policy_net(state)
      if self.action_space_is_discrete:
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        action = raw_action.max(1).indices.view(1, 1).item()
      else:   # Continuous
        action = raw_action.cpu().detach().numpy()[0]
      #print(f"get_policy_action: action = {action} {type(action)}")
      return action


  def get_epsilon(self):
    return self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_done / self.EPS_DECAY)
    
  def select_action(self,state):
    sample = random.random()
    eps_threshold = self.get_epsilon()
    self.steps_done += 1
    if sample > eps_threshold:
      #print("select_action: policy action")
      return self.get_policy_action(state)
    else:
      sample = self.env.env.action_space.sample()
      #print(f"select_action: sample = {sample} {type(sample)}")
      return sample

  def plot_progress(self, block = False):
      fig = plt.figure(num=1)
      plt.clf()
      gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])  # 3 rows, height ratio 3:1:1

      # Plot episode durations by episode
      ax1 = fig.add_subplot(gs[0])  
      ax1.set_xlabel('Episode')
      #ax1.set_ylabel('Duration')
      #values_to_plot = torch.tensor(self.meta_state.get_values("episode_durations"), dtype=torch.float)
      ax1.set_ylabel('Reward Total')
      values_to_plot = torch.tensor(self.meta_state.get_values("reward_total"), dtype=torch.float)
      ax1.plot(values_to_plot.numpy())
      
      # Draw a smoothed graph
      smooth_size = 30
      if len(values_to_plot) >= smooth_size:
          smoothed = values_to_plot.unfold(0, smooth_size, 1).mean(1).view(-1)
          #smoothed = torch.cat((torch.zeros(smooth_size-1), smoothed))
          smoothed = smoothed.numpy()
          self.average_duration = smoothed[-1]
          ax1.plot(np.arange(smooth_size//2, smooth_size//2+len(smoothed)),smoothed)
          
      # Plot epsilon by episode
      ax2 = fig.add_subplot(gs[1])  
      ax2.set_ylabel('Epsilon')
      ax2.plot(self.meta_state.get_values("epsilon"))

      # Plot memory size by episode
      ax3 = fig.add_subplot(gs[2])  
      ax3.set_ylabel('Memory size')
      ax3.plot(self.meta_state.get_values("memory_size"))

      plt.pause(0.01)  # pause a bit so that plots are updated
      plt.show(block=block)
   
  def optimize_model(self):
      if len(self.memory) < self.BATCH_SIZE:
          return
      transitions = self.memory.sample(self.BATCH_SIZE)
      # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
      # detailed explanation). This converts batch-array of Transitions
      # to Transition of batch-arrays.
      batch = Transition(*zip(*transitions))

      # Compute a mask of non-final states and concatenate the batch elements
      # (a final state would've been the one after which simulation ended)
      non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
      non_final_next_states = torch.cat([s for s in batch.next_state
                                                  if s is not None])
      state_batch = torch.cat(batch.state)
      action_batch = torch.cat(batch.action)
      reward_batch = torch.cat(batch.reward)

      # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
      # columns of actions taken. These are the actions which would've been taken
      # for each batch state according to policy_net
      state_action_values = self.policy_net(state_batch).gather(1, action_batch)

      # Compute V(s_{t+1}) for all next states.
      # Expected values of actions for non_final_next_states are computed based
      # on the "older" target_net; selecting their best reward with max(1).values
      # This is merged based on the mask, such that we'll have either the expected
      # state value or 0 in case the state was final.
      next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
      with torch.no_grad():
          next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
      # Compute the expected Q values
      expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

      # Compute Huber loss
      criterion = nn.SmoothL1Loss()
      loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

      # Optimize the model
      self.optimizer.zero_grad()
      loss.backward()
      # In-place gradient clipping
      torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
      self.optimizer.step()

  def update_target_net_from_policy_net(self):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
    self.target_net.load_state_dict(target_net_state_dict)
    
  def do_episode(self):
    # Initialize the environment and get its state
    state, info = self.env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    reward_total = 0
    
    for steps in count():
        action = self.select_action(state)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        reward_total += reward.cpu().item()
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Store the transition in memory
        action_tensor = torch.tensor([[action]], device=self.device, dtype=torch.long)
        #print(f"do_episode: action = {action} {type(action)}, action_tensor = {action_tensor} {type(action_tensor)}")
        self.memory.push(state, action_tensor, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update target net
        self.update_target_net_from_policy_net()

        if done:
          self.episode_ended(steps, reward_total)
          break
    return steps

  def loop_episodes(self,num_episodes, visualize_every=0):
    i_episode = self.meta_state.get_latest_value("episodes")
    while i_episode < num_episodes :
        i_episode += 1
        #if self.average_duration > 450:  # early out of episodes because its already good enough
        #  break
        
        steps = self.do_episode()
        
        # #extra training between episodes
        # print(f"Extra training {steps*5} times")
        # for i in range(steps*5):
          # self.optimize_model()
          # self.update_target_net_from_policy_net()
                  
        if visualize_every != 0:
          if i_episode % visualize_every == 0:
            self.visualize_model(1)

  def episode_ended(self, steps, reward_total):
    self.meta_state.add_value("episodes",self.meta_state.get_latest_value("episodes") + 1)
    self.meta_state.add_value("steps_done",self.steps_done)
    self.meta_state.add_value("epsilon",self.get_epsilon())
    self.meta_state.add_value("memory_size",len(self.memory))
    self.meta_state.add_value("episode_durations",steps + 1)
    self.meta_state.add_value("reward_total",reward_total)

    if self.meta_state.get_latest_value("episodes")%5 == 0:  # save every 5 episodes to avoid slowing things down too much
      self.save()
    self.plot_progress()
    

  def visualize_model(self,num_episodes = 5):
    #env_visualize = gym.make(self.gym_name, render_mode="human")  # Use "human" for visualization
    env_visualize = self.creat_env_fn(render_mode="human")  # Use "human" for visualization

    for i_episode in range(num_episodes):
      state, info = env_visualize.reset()
      state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
      for t in count():
        env_visualize.render()  # Render the environment
        action = self.get_policy_action(state)
        observation, reward, terminated, truncated, _ = env_visualize.step(action)
        
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if terminated or truncated:
          print(f"Episode {i_episode + 1} finished after {t + 1} timesteps")
          break
    env_visualize.close()

  def visualize_model_hardwired_cartpole(self,num_episodes = 5):
    #env_visualize = gym.make(self.gym_name, render_mode="human")  # Use "human" for visualization
    env_visualize = self.creat_env_fn(render_mode="human")  # Use "human" for visualization

    for i_episode in range(num_episodes):
      state, info = env_visualize.reset()
      state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
      for t in count():
        env_visualize.render()  # Render the environment
        
        #action = self.policy_net(state).max(1).indices.view(1, 1)
        #observation, reward, terminated, truncated, _ = env_visualize.step(action.item())

        # custom hand-written control code
        # extract state into human readable form
        cart_position       = state[0].cpu().numpy()[0]
        cart_velocity       = state[0].cpu().numpy()[1]
        pole_angle          = state[0].cpu().numpy()[2]
        pole_angle_velocity = state[0].cpu().numpy()[3]

        # # simple minded move left if tilting left, move right if tilting right
        # action = np.array([ 0 if pole_angle<0 else 1 ]) 

        # account for angle and angle velocity (worked pretty well)
        angle_comb = pole_angle + pole_angle_velocity
        action = np.array([ 0 if angle_comb<0 else 1 ]) 

        observation, reward, terminated, truncated, _ = env_visualize.step(action.item())
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if terminated or truncated:
          print(f"Episode {i_episode + 1} finished after {t + 1} timesteps")
          break
    env_visualize.close()
