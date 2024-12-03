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
		self.ac_kwargs=dict()
		self.seed=0, 
		self.steps_per_epoch=4000
		self.epochs=100
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
		self.max_ep_len=1000 
		self.logger_kwargs=dict()
		self.save_freq=1
		
		self.epoch = 0
		self.steps_per_epoch_cursor = self.steps_per_epoch
		self.av_test_episode_reward = -500	# TODO - find a better way of determining the lowest reasonable test score
		
		self.actor = core.MLPActorCritic(self.create_env_fn, hidden_layer_sizes=self.settings["hidden_layer_sizes"], learning_rate=self.lr)
		self.test_env = self.create_env_fn()
		obs_dim = self.env.observation_space.shape
		act_dim = self.env.action_space.shape[0]

		# # Action limit for clamping: critically, assumes all dimensions share the same bound!
		# act_limit = env.action_space.high[0]

		# Create actor-critic module and target networks
		#ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
		self.target_actor = deepcopy(self.actor)

		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.target_actor.parameters():
			p.requires_grad = False
			
		# List of parameters for both Q-networks (save this for convenience)
		self.q_params = itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters())

		# Experience buffer
		#self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.replay_size)
		self.memory = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.replay_size)

		# Count variables (protip: try to get a feel for how different size networks behave!)
		var_counts = tuple(core.count_vars(module) for module in [self.actor.pi, self.actor.q1, self.actor.q2])
		print('Number of parameters: \t pi: %d, \t q1: %d, \t q2: %d'%var_counts)

		# Set up optimizers for policy and q-function
		self.pi_optimizer = Adam(self.actor.pi.parameters(), lr=self.lr)
		self.q_optimizer = Adam(self.q_params, lr=self.lr)

		# Set up model saving
		#logger.setup_pytorch_saver(ac)

		self.data_to_plot = ["episode_reward","av_test_episode_reward","last_step_reward","episode_durations","memory_size"]
		self.load_if_save_exists()

	def add_data_to_save(self):
		super().add_data_to_save()
		self.saver.add_data_to_save( "actor",				self.actor)
		self.saver.add_data_to_save( "target_actor",		self.target_actor)
	
	def load(self):
		super().load()
		self.saver.load_data_into( "actor",				self.actor)
		self.saver.load_data_into( "target_actor",		self.target_actor)
		self.steps_per_epoch_cursor 	= self.logger.get_latest_value("steps_per_epoch_cursor")
		self.epoch						= self.logger.get_latest_value("epoch")
		self.av_test_episode_reward		= self.logger.get_latest_value("av_test_episode_reward")
		self.test_agent()

		# List of parameters for both Q-networks (save this for convenience)
		self.q_params = itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters())
		# Set up optimizers for policy and q-function
		self.pi_optimizer = Adam(self.actor.pi.parameters(), lr=self.lr)
		self.q_optimizer = Adam(self.q_params, lr=self.lr)

	def episode_ended(self, last_step_reward, steps, episode_reward):
		self.steps_per_epoch_cursor -= steps
		if self.steps_per_epoch_cursor <= 0:
			self.steps_per_epoch_cursor = self.steps_per_epoch
			self.epoch_ended()
		
		self.logger.set_frame_value("steps_per_epoch_cursor",		self.steps_per_epoch_cursor)
		self.logger.set_frame_value("epoch",						self.epoch)
		self.logger.set_frame_value("av_test_episode_reward",		self.av_test_episode_reward)

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

		# Useful info for logging
		pi_info = dict(LogPi=logp_pi.detach().numpy())

		return loss_pi, pi_info
	
	def update(self, data):
		# First run one gradient descent step for Q1 and Q2
		self.q_optimizer.zero_grad()
		loss_q, q_info = self.compute_loss_q(data)
		loss_q.backward()
		self.q_optimizer.step()

		# Record things
		#logger.store(LossQ=loss_q.item(), **q_info)

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

		# Record things
		#logger.store(LossPi=loss_pi.item(), **pi_info)

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

	def test_agent(self):
		print("test_agent")
		test_episode_rewards = []
		for j in range(self.num_test_episodes):
			observation, info = self.test_env.reset()
			terminated, episode_reward, steps = False, 0, 0
			while not(terminated or (steps == self.max_ep_len)):
				# Take deterministic actions at test time 
				observation, reward, terminated, truncated, _ = self.test_env.step(self.get_action(observation, True))
				episode_reward += reward
				steps += 1
			print(f"Test episode {j}: steps = {steps+1}, episode_reward = {episode_reward:0.1f}, last step reward = {reward:0.1f}") 
			test_episode_rewards.append(episode_reward)
		self.av_test_episode_reward = average = np.mean(np.array(test_episode_rewards))
		print(f"av_test_episode_reward = {self.av_test_episode_reward:0.1f}")

	def do_episode(self):
		observation, info = self.env.reset()
		self.actor.reset()
		#observation_tensor = self.actor.to_tensor_observation(observation)
		episode_reward = 0
		
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

			# Ignore the "done" signal if it comes from hitting the time
			# horizon (that is, when it's an artificial terminal signal
			# that isn't based on the agent's state)
			done = False if steps==self.max_ep_len else terminated

			# Store experience to replay buffer
			self.memory.store(observation, action, reward, observation2, done)

			# Super critical, easy to overlook step: make sure to update 
			# most recent observation!
			observation = observation2

			# end of episode handling
			self.steps_done += 1
			if done or (steps == self.max_ep_len):
				self.episode_ended_outer(last_step_reward=reward, steps=steps, episode_reward=episode_reward)
				break

			# Update handling
			if self.steps_done >= self.update_after and self.steps_done % self.update_every == 0:
				for j in range(self.update_every):
					batch = self.memory.sample_batch(self.batch_size)
					self.update(data=batch)
			
		return reward, steps, episode_reward

	def epoch_ended(self):
		self.epoch += 1
		print(f"Epoch {self.epoch}")

		# Save model
		#if (epoch % save_freq == 0) or (epoch == epochs):
			#logger.save_state({'env': env}, None)

		# Test the performance of the deterministic version of the agent.
		self.test_agent()

		# Log info about epoch
		#logger.log_tabular('Epoch', epoch)
		#logger.log_tabular('EpRet', with_min_and_max=True)
		#logger.log_tabular('TestEpRet', with_min_and_max=True)
		#logger.log_tabular('EpLen', average_only=True)
		#logger.log_tabular('TestEpLen', average_only=True)
		#logger.log_tabular('TotalEnvInteracts', t)
		#logger.log_tabular('Q1Vals', with_min_and_max=True)
		#logger.log_tabular('Q2Vals', with_min_and_max=True)
		#logger.log_tabular('LogPi', with_min_and_max=True)
		#logger.log_tabular('LossPi', average_only=True)
		#logger.log_tabular('LossQ', average_only=True)
		#logger.log_tabular('Time', time.time()-start_time)
		#logger.dump_tabular()
		# Log info about epoch
		


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	#parser.add_argument('--env', type=str, default='HalfCheetah-v5')
	parser.add_argument('--env', type=str, default='LunarLanderContinuous-v3')
	parser.add_argument('--hid', type=int, default=256)
	parser.add_argument('--l', type=int, default=2)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--seed', '-s', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--exp_name', type=str, default='sac')
	args = parser.parse_args()

	#from spinup.utils.run_utils import setup_logger_kwargs
	#logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	torch.set_num_threads(torch.get_num_threads())

	sac(lambda **kwargs : gym.make(args.env,**kwargs), actor_critic=core.MLPActorCritic,
		ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
		gamma=args.gamma, seed=args.seed, epochs=args.epochs,
		#logger_kwargs=logger_kwargs)
		)
