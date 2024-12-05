# https://github.com/createamind/DRL/blob/master/spinup/algos/sac1/sac1_LunarLanderContinuous-v2_100ep.py
# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/haarnoja/sac

import pathlib
algo_name = pathlib.Path(__file__).stem

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import algos.core as core
from itertools import count

class ReplayBuffer:
	"""
	A simple FIFO experience replay buffer for SAC agents.
	"""

	def __init__(self, obs_dim, act_dim, size):
		self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
		self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
		self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
		self.rew_buf = np.zeros(size, dtype=np.float32)
		self.done_buf = np.zeros(size, dtype=np.float32)
		self.ptr, self.size, self.max_size = 0, 0, size

	def store(self, obs, act, rew, next_obs, done):
		self.obs_buf[self.ptr] = obs
		self.obs2_buf[self.ptr] = next_obs
		self.act_buf[self.ptr] = act
		self.rew_buf[self.ptr] = rew
		self.done_buf[self.ptr] = done
		self.ptr = (self.ptr+1) % self.max_size
		self.size = min(self.size+1, self.max_size)

	def sample_batch(self, batch_size=32):
		idxs = np.random.randint(0, self.size, size=batch_size)
		batch = dict(obs=self.obs_buf[idxs],
					 obs2=self.obs2_buf[idxs],
					 act=self.act_buf[idxs],
					 rew=self.rew_buf[idxs],
					 done=self.done_buf[idxs])
		return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

	def __len__(self):
		return self.size
	
	def to_saveable(self):
		return [ self.obs_buf,
				 self.obs2_buf,
				 self.act_buf,
				 self.rew_buf,
				 self.done_buf,
				 self.ptr, self.size, self.max_size]

		
	def from_saveable(self, saveable):
		self.obs_buf = saveable[0]
		self.obs2_buf = saveable[1]
		self.act_buf = saveable[2]
		self.rew_buf = saveable[3]
		self.done_buf = saveable[4]
		self.ptr = saveable[5]
		self.size = saveable[6]
		self.max_size = saveable[7]


class Algo(core.AlgoBase):
	def __init__(self, create_env_fn, settings=[]):
		super().__init__(name=algo_name, create_env_fn=create_env_fn, settings=settings)

		# sac parameters
		self.replay_size=int(1e6)
		self.gamma=0.99
		self.polyak=0.995
		self.lr=1e-3
		self.alpha=0.2
		self.batch_size=100
		self.start_steps=10000
		self.update_after=1000
		self.update_every=50
		self.num_test_episodes=10
		self.max_ep_steps = 200
		
		self.actor = core.MLPActorCritic(self.create_env_fn, hidden_layer_sizes=self.settings["hidden_layer_sizes"], learning_rate=self.lr)
		self.test_env = self.create_env_fn()
		self.target_actor = deepcopy(self.actor)

		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.target_actor.parameters():
			p.requires_grad = False
			
		# List of parameters for both Q-networks (save this for convenience)
		self.q_params = itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters())

		# Experience buffer
		obs_dim = self.env.observation_space.shape
		act_dim = self.env.action_space.shape[0]
		self.memory = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.replay_size)

		# Count variables (protip: try to get a feel for how different size networks behave!)
		var_counts = tuple(core.count_vars(module) for module in [self.actor.pi, self.actor.q1, self.actor.q2])
		print('Number of parameters: \t pi: %d, \t q1: %d, \t q2: %d'%var_counts)

		# Set up optimizers for policy and q-function
		self.pi_optimizer = Adam(self.actor.pi.parameters(), lr=self.lr)
		self.q_optimizer = Adam(self.q_params, lr=self.lr)

		self.data_to_plot = [["episode_reward","recent_test_av"],"loss_q","loss_pi","last_step_reward","episode_durations","memory_size"]
		self.load_if_save_exists()

	def add_data_to_save(self):
		super().add_data_to_save()
		self.saver.add_data_to_save( "actor",				self.actor)
		self.saver.add_data_to_save( "target_actor",		self.target_actor)
	
	def load(self):
		super().load()
		self.saver.load_data_into( "actor",				self.actor)
		self.saver.load_data_into( "target_actor",		self.target_actor)
		#self.actor.test(create_env_fn=self.create_env_fn)

	def episode_ended(self, last_step_reward, steps, episode_reward):
		super().episode_ended(last_step_reward, steps, episode_reward)		# do all the standard stuff at the end of an episode

	# Set up function for computing SAC Q-losses
	def compute_loss_q(self, data):
		o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

		q1 = self.actor.q1(o,a)
		q2 = self.actor.q2(o,a)

		# Bellman backup for Q functions
		with torch.no_grad():
			# Target actions come from *current* policy
			a2, logp_a2 = self.actor.pi(o2)

			# Target Q-values
			q1_pi_targ = self.target_actor.q1(o2, a2)
			q2_pi_targ = self.target_actor.q2(o2, a2)
			q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
			backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

		# MSE loss against Bellman backup
		loss_q1 = ((q1 - backup)**2).mean()
		loss_q2 = ((q2 - backup)**2).mean()
		loss_q = loss_q1 + loss_q2
		self.logger.set_frame_value("loss_q", loss_q.item())

		# Useful info for logging
		q_info = dict(Q1Vals=q1.detach().numpy(),
					  Q2Vals=q2.detach().numpy())

		return loss_q, q_info

	# Set up function for computing SAC pi loss
	def compute_loss_pi(self, data):
		o = data['obs']
		pi, logp_pi = self.actor.pi(o)
		q1_pi = self.actor.q1(o, pi)
		q2_pi = self.actor.q2(o, pi)
		q_pi = torch.min(q1_pi, q2_pi)

		# Entropy-regularized policy loss
		loss_pi = (self.alpha * logp_pi - q_pi).mean()
		self.logger.set_frame_value("loss_pi", loss_pi.item())

		# Useful info for logging
		pi_info = dict(LogPi=logp_pi.detach().numpy())

		return loss_pi, pi_info
	
	def update(self, data):
		# First run one gradient descent step for Q1 and Q2
		self.q_optimizer.zero_grad()
		loss_q, q_info = self.compute_loss_q(data)
		loss_q.backward()
		self.q_optimizer.step()

		# Freeze Q-networks so you don't waste computational effort 
		# computing gradients for them during the policy learning step.
		for p in self.q_params:
			p.requires_grad = False

		# Next run one gradient descent step for pi.
		self.pi_optimizer.zero_grad()
		loss_pi, pi_info = self.compute_loss_pi(data)
		loss_pi.backward()
		self.pi_optimizer.step()

		# Unfreeze Q-networks so you can optimize it at next DDPG step.
		for p in self.q_params:
			p.requires_grad = True

		# Finally, update target networks by polyak averaging.
		with torch.no_grad():
			for p, p_targ in zip(self.actor.parameters(), self.target_actor.parameters()):
				# NB: We use an in-place operations "mul_", "add_" to update target
				# params, as opposed to "mul" and "add", which would make new tensors.
				p_targ.data.mul_(self.polyak)
				p_targ.data.add_((1 - self.polyak) * p.data)

	def get_action(self, o, deterministic=False):
		return self.actor.act(torch.as_tensor(o, dtype=torch.float32), 
					  deterministic)

	def do_episode(self):
		observation, info = self.env.reset()
		self.actor.reset()
		episode_reward = 0
		start_time = time.time()
		
		for steps in count():
			# Until start_steps have elapsed, randomly sample actions
			# from a uniform distribution for better exploration. Afterwards, 
			# use the learned policy. 
			if self.steps_done > self.start_steps:
				action = self.get_action(observation)
			else:
				action = self.env.action_space.sample()

			# Step the env
			observation2, reward, terminated, truncated, _ = self.env.step(action)
			episode_reward += reward
			done = truncated or terminated or (steps > self.max_ep_steps)

			# Store experience to replay buffer
			self.memory.store(observation, action, reward, observation2, done)

			# Super critical, easy to overlook step: make sure to update 
			# most recent observation!
			observation = observation2

			# end of episode handling
			self.steps_done += 1
			if done:
				self.episode_ended_outer(last_step_reward=reward, steps=steps+1, episode_reward=episode_reward, time_taken=time.time() - start_time)
				break

			# Update handling
			if self.steps_done >= self.update_after and self.steps_done % self.update_every == 0:
				for j in range(self.update_every):
					batch = self.memory.sample_batch(self.batch_size)
					self.update(data=batch)
			
		return reward, steps, episode_reward

