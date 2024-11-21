import algos.core as core
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from itertools import count

class Algo(core.AlgoBase):
	def __init__(self, create_env_fn, settings=[]):
		super().__init__(name="dqn", create_env_fn=create_env_fn, settings=settings)
		self.settings = core.Saveable(settings)
		self.actor = core.MLPActorDiscreteActions(self.create_env_fn, hidden_layer_sizes=self.settings["hidden_layer_sizes"])
		self.optimizer = optim.AdamW(self.actor.mlp.parameters(), lr=self.LR, amsgrad=True)
		self.target_actor = self.actor.create_copy()
		self.load_if_save_exists()

	def save(self):
		self.saver.add_data_to_save( "actor",			self.actor.mlp, 		is_net=True )
		self.saver.add_data_to_save( "target_actor",	self.target_actor.mlp, 	is_net=True )
		self.saver.add_data_to_save( "memory",			self.memory )
		self.saver.add_data_to_save( "settings", 		self.settings )
		self.saver.add_data_to_save( "logger",			self.logger )
		self.saver.save()
	
	def load(self):
		self.saver.load_data_into( "actor",				self.actor.mlp, 		is_net=True )
		self.saver.load_data_into( "target_actor",		self.target_actor.mlp, 	is_net=True )
		self.saver.load_data_into( "memory",			self.memory )
		self.saver.load_data_into( "settings", 			self.settings )
		self.saver.load_data_into( "logger", 			self.logger )
		self.post_load_fixup();
	
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
		self.optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.actor.mlp.parameters(), 100)
		self.optimizer.step()

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
		this_episode = self.logger.get_latest_value("episodes") + 1
		self.logger.set_frame_value("episodes",				this_episode)
		self.logger.set_frame_value("steps_done",			self.steps_done)
		self.logger.set_frame_value("epsilon",				self.get_epsilon())
		self.logger.set_frame_value("memory_size",			len(self.memory))
		self.logger.set_frame_value("episode_durations",	steps + 1)
		self.logger.set_frame_value("episode_reward",		episode_reward)
		self.logger.set_frame_value("last_step_reward",		last_step_reward)
		self.logger.next_frame()
		print(f"episode_ended {this_episode}: steps = {steps+1}, episode_reward = {episode_reward:0.1f}, last step reward = {last_step_reward:0.1f}") 

		if self.logger.get_latest_value("episodes")%5 == 0:	# save every 5 episodes to avoid slowing things down too much
			self.save()
			
	def do_episode_test(self, seed=None):
		# Initialize the environment and get its observation
		observation, info = self.env.reset(seed=seed)
		observation_tensor = self.actor.to_tensor_observation(observation)
		episode_reward = 0
		
		for steps in count():
			action = self.actor.select_action(observation_tensor)	# always use the on policy action when testing
			observation, reward, terminated, truncated, _ = self.env.step(action)
			episode_reward += reward
			reward_tensor = self.actor.to_tensor_reward(reward)
			done = terminated or truncated
			if terminated:
				next_observation_tensor = None
			else:
				next_observation_tensor = self.actor.to_tensor_observation(observation)
			observation_tensor = next_observation_tensor
			if done:
				print(f"test episode ended: steps = {steps+1}, episode_reward = {episode_reward:0.1f}, last step reward = {reward:0.1f}")
				break
		return reward, steps, episode_reward

	def show_graph(self):
		self.logger.plot(data_to_plot=["episode_reward","last_step_reward","episode_durations","episodes","memory_size","epsilon","steps_done"])

	def loop_episodes(self,num_episodes, visualize_every=0, show_graph=True):
		i_episode = self.logger.get_latest_value("episodes")
		while i_episode < num_episodes :
			i_episode += 1
			last_step_reward, steps, episode_reward, episode_recording = self.do_episode()
			if show_graph:
				self.show_graph()
			# if last_step_reward >= 100:	# TODO: for this to work, we'll need the environment to be deterministic, so will need to record the random seed
				# print(f"As last_step_reward = {last_step_reward}, visualize episode replay")
				# self.actor.visualize_model_from_recording(self.create_env_fn, episode_recording)
			if visualize_every != 0:
				if i_episode % visualize_every == 0:
					self.visualize(num_episodes = 1)

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
	