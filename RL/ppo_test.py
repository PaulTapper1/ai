# code taken from https://discuss.pytorch.org/t/very-simple-environment-with-continuous-action-space-fails-to-learn-effectively-with-ppo/182397
# 
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import ppo

# == Hyperparameters
EPISODE_LEN = 300
TRAINING_LEN = int(1e7)  # Absolute upper bound on training

ACTION_STD = 0.6  # Stating standard distribution for actions
# Linear decay of standard distribution means
# eventually picking actions closer to the mean
ACTION_STD_DECAY_RATE = 0.05
MIN_ACTION_STD = 0.1  # Prevent decay beyond this
ACTION_STD_DECAY_FREQ = int(2e5)  # Decay every so many time-steps

UPDATE_STEP = EPISODE_LEN * 5  # Also replay memory size
APPEND_FREQ = 1
K_EPOCHS = 80

EPS_CLIP = 0.2  # Usual choice
GAMMA = 0.99

LR_CRITIC = 10e-4
LR_ACTOR = 3e-4

#gym_name = 'LunarLanderContinuous-v3'
gym_name = "MountainCarContinuous-v0" # https://github.com/openai/gym/wiki/MountainCarContinuous-v0

env = gym.make(gym_name)
state, info = env.reset()
OBS_DIM = len(state)
ACT_DIM = env.action_space.shape[0]

SAVE_FREQ = 100_000
PRINT_FREQ = 250

# == Training


ppo_agent = ppo.PPO(
    OBS_DIM, ACT_DIM,
    LR_ACTOR, LR_CRITIC,
    GAMMA, K_EPOCHS, EPS_CLIP,
    ACTION_STD
)

def run(episodes, agent: ppo.PPO):
    env_visualise = gym.make(gym_name, render_mode="human")
    for _ in range(episodes):
        obs, _ = env_visualise.reset()
        done = False

        step = 0
        total_reward = 0

        while not done:
            action = agent.select_action(obs)

            obs, reward, term, trunc, info = env_visualise.step(action)
            total_reward += reward

            env_visualise.render()
            print('\r', f'Reward {reward:.2f}; Total reward {total_reward:.2f}; (step {step}) by Action {action}')

            done = term or trunc
            step += 1

def update_graph(values):
  fig = plt.figure(num=1)
  plt.clf()
  gs = gridspec.GridSpec(1, 1)
  #gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])  # 4 rows, height ratio 3:1:1:1
  subplot = -1
  fontsize = 9

  # Plot episode durations by episode
  subplot += 1
  ax = fig.add_subplot(gs[subplot])  
  ax.set_xlabel('Episode', fontsize=fontsize)
  #ax1.set_ylabel('Duration')
  #values_to_plot = torch.tensor(self.meta_state.get_values("episode_durations"), dtype=torch.float)
  ax.set_ylabel('Reward Total', fontsize=fontsize)
  ax.plot(values)  

  plt.pause(0.01)  # pause a bit so that plots are updated
  plt.show(block=False)


graph_total_reward = []

step = 0
episode = 0
while step < TRAINING_LEN:
    episode += 1
    obs, _ = env.reset()
    term = trunc = False

    total_reward: int = 0
    for t in range(10000):
        step += 1
        # Select action
        action = ppo_agent.select_action(obs)
        # Take action and observe change and compute reward
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        # Save reward and terminal state
        if step % APPEND_FREQ == 0:
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(term or trunc)
        # Potentially update the agent networks
        if step % UPDATE_STEP == 0:
            print('\nUpdating agent networks')
            ppo_agent.update()
        # Decay standard deviation
        if step % ACTION_STD_DECAY_FREQ == 0:
            print('\nDecaying action STD')
            ppo_agent.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)

        #if step % PRINT_FREQ == 0:
        #    print(f'\rEpisode: {episode}; Step {t} ({step}); Reward {reward} (Total {total_reward})', end='')

        if step % SAVE_FREQ == 0:
            ppo_agent.save('./ppo_agent_checkpoint.pth')

        if term: print(f'Terminated (Eps: {episode}; in {t} steps) (Reward: {reward} / {total_reward})')
        # Terminate episode if done
        if term or trunc: break
    
    graph_total_reward.append(total_reward)
    update_graph(graph_total_reward)
    

print('')

wait: str = input('run sim? (y/n): ')
if wait.lower().strip() == 'n':
    exit(0)

# Run fully trained network
run(5, ppo_agent)