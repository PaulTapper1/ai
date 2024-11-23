import algos.core as core
import torch
import torch.nn as nn
import math
import random
from itertools import count
import pathlib
algo_name = pathlib.Path(__file__).stem

class Algo(core.AlgoBase):
	def __init__(self, create_env_fn, settings=[]):
		super().__init__(name=algo_name, create_env_fn=create_env_fn, settings=settings)
		self.actor = core.MLPActorDiscreteActions(self.create_env_fn, hidden_layer_sizes=self.settings["hidden_layer_sizes"], learning_rate=self.LR)
		self.target_actor = self.actor.create_copy()
		self.load_if_save_exists()

	def add_data_to_save(self):
		super().add_data_to_save()
		self.saver.add_data_to_save( "actor",			self.actor.mlp, 		is_net=True )
		self.saver.add_data_to_save( "target_actor",	self.target_actor.mlp, 	is_net=True )
	
	def load(self):
		super().load()
		self.saver.load_data_into( "actor",				self.actor.mlp, 		is_net=True )
		self.saver.load_data_into( "target_actor",		self.target_actor.mlp, 	is_net=True )
	
	def get_epsilon(self):
		decay = self.logger.get_latest_value("episodes")		 # epsilon based on episodes
		return self.EPS_END + (self.EPS_START - self.EPS_END) * \
			math.exp(-1. * decay / self.EPS_DECAY)

	def select_action(self, observation):
		eps_threshold = self.get_epsilon()
		self.steps_done += 1
		if random.random() > eps_threshold:
			action = self.actor.select_action(observation)
			return action
		else:
			random_action = self.env.action_space.sample()
			return random_action

	def optimize_model(self):
		if len(self.memory) < self.BATCH_SIZE:
				return
		transitions = self.memory.sample(self.BATCH_SIZE)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = core.Transition(*zip(*transitions))

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
		state_action_values = self.actor.mlp(state_batch).gather(1, action_batch)
		
		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_actor; selecting their best reward with max(1).values
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
		with torch.no_grad():
			next_state_values[non_final_mask] = self.target_actor.mlp(non_final_next_states).max(1).values
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

		# Compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
			
		# Optimize the model
		self.actor.optimize(loss=loss, clip_grad_value=100)
		# self.optimizer.zero_grad()
		# loss.backward()
		# # In-place gradient clipping
		# torch.nn.utils.clip_grad_value_(self.actor.mlp.parameters(), 100)
		# self.optimizer.step()

	def update_target_actor_from_policy_net(self):
		# Soft update of the target network's weights
		# θ′ ← τ θ + (1 −τ )θ′
		target_actor_state_dict = self.target_actor.mlp.state_dict()
		policy_net_state_dict = self.actor.mlp.state_dict()
		for key in policy_net_state_dict:
				target_actor_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_actor_state_dict[key]*(1-self.TAU)
		self.target_actor.mlp.load_state_dict(target_actor_state_dict)
		
	def do_episode(self):
		# Initialize the environment and get its observation
		observation, info = self.env.reset()
		observation_tensor = self.actor.to_tensor_observation(observation)
		episode_reward = 0
		episode_recording = []
		
		for steps in count():
			action = self.select_action(observation_tensor)
			
			episode_recording.append(action)
			observation, reward, terminated, truncated, _ = self.env.step(action)
			episode_reward += reward
			reward_tensor = self.actor.to_tensor_reward(reward)
			done = terminated or truncated

			if terminated:
				next_observation_tensor = None
			else:
				next_observation_tensor = self.actor.to_tensor_observation(observation)

			# Store the transition in memory
			action_tensor = self.actor.to_tensor_action(action)
			self.memory.push(observation_tensor, action_tensor, next_observation_tensor, reward_tensor)

			# Move to the next observation
			observation_tensor = next_observation_tensor

			# Perform one step of the optimization (on the policy network)
			self.optimize_model()

			# Soft update target net
			self.update_target_actor_from_policy_net()

			if done:
				self.episode_ended(last_step_reward=reward, steps=steps, episode_reward=episode_reward)
				break
		return reward, steps, episode_reward, episode_recording

	def episode_ended(self, last_step_reward, steps, episode_reward):
		# log out any algorithm specific data you want to track
		self.logger.set_frame_value("epsilon",				self.get_epsilon())
		super().episode_ended(last_step_reward, steps, episode_reward)		# do all the standard stuff at the end of an episode

	def show_graph(self):
		self.logger.plot(data_to_plot=["episode_reward","last_step_reward","episode_durations","memory_size","epsilon"])

#####################################################################################
# for testing
if __name__ == '__main__':
	import gymnasium as gym
	
	print("Running code tests...")
	
	print("\nTesting Algo...")
	settings = { "hidden_layer_sizes" : [128,64,32] }
	def create_env_fn_LunarLander(render_mode=None):
		return gym.make("LunarLander-v3",render_mode=render_mode)		# https://gymnasium.farama.org/environments/box2d/lunar_lander/
	algo = Algo( create_env_fn_LunarLander, settings=settings )	
	#algo.visualize()	# test visualization of actor in environment
	#algo.loop_episodes(1000, visualize_every=10, show_graph=True)
	algo.loop_episodes(1000, visualize_every=0, show_graph=True)
	