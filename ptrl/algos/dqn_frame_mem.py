import algos.core as core
import torch
import torch.nn as nn
import algos.dqn as dqn
import random
from itertools import count
import pathlib
algo_name = pathlib.Path(__file__).stem

class MLPActorDiscreteActionsFrameMem(core.MLPActor):
	def __init__(self, create_env_fn, hidden_layer_sizes=[256,256],
				 activation=nn.ReLU, output_activation=nn.Identity, **kwargs):
		temp_env = create_env_fn()
		if not f"{temp_env.action_space}".startswith("Discrete"):
			core.print_error("Cannot create an MLPActorDiscreteActionsFrameMem using a continuous action space")
		action_dim = temp_env.action_space.n
		state, info = temp_env.reset()
		observation_dim = len(state)
		self.action_dim = action_dim
		self.frame_mem_dim = observation_dim+action_dim
		temp_env.close()
		super().__init__(observation_dim=(observation_dim+self.frame_mem_dim), action_dim=action_dim,
						 hidden_layer_sizes=hidden_layer_sizes, activation=activation, **kwargs)
		self.reset()

	def reset(self):
		self.frame_mem = torch.tensor([[0]*self.frame_mem_dim], device=self.device, dtype=torch.float32)
	
	def combine_frame_mem_and_observation(self, observation):
		return torch.cat( (self.frame_mem[0], observation[0]) ).unsqueeze(0)
	
	def select_action(self, observation):
		# combine this frame's observation with the frame_mem of the observation and action from the previous frame
		combined_observation = self.combine_frame_mem_and_observation(observation)
		raw_action = super().select_action(combined_observation)

		# update the frame_mem with the latest observation and selected action
		self.frame_mem = torch.cat( (observation[0], raw_action[0]) ).unsqueeze(0)

		# t.max(1) will return the largest column value of each row.
		# second column on max result is index of where max element was
		# found, so we pick action with the larger expected reward.
		action = raw_action.max(1).indices.view(1, 1).item()
		return action
		
	def store_frame_mem(self, observation, action):
		raw_action_array = [0]*self.action_dim
		raw_action_array[action] = 1
		raw_action = torch.tensor([raw_action_array], device=self.device, dtype=torch.float32)
		self.frame_mem = torch.cat( (observation[0], raw_action[0]) ).unsqueeze(0)		

class Algo(dqn.Algo):
	def __init__(self, create_env_fn, settings=[]):
		core.AlgoBase.__init__(self, name=algo_name, create_env_fn=create_env_fn, settings=settings)
		self.actor = MLPActorDiscreteActionsFrameMem(self.create_env_fn, hidden_layer_sizes=self.settings["hidden_layer_sizes"], learning_rate=self.LR)
		self.target_actor = self.actor.create_copy()
		self.load_if_save_exists()

	def select_action(self, observation):
		eps_threshold = self.get_epsilon()
		self.steps_done += 1
		if random.random() > eps_threshold:
			action = self.actor.select_action(observation)
			return action
		else:
			random_action = self.env.action_space.sample()
			self.actor.store_frame_mem(observation, random_action)
			return random_action

	def do_episode(self):
		# Initialize the environment and get its observation
		observation, info = self.env.reset()
		observation_tensor = self.actor.to_tensor_observation(observation)
		observation_tensor_combined = self.actor.combine_frame_mem_and_observation(observation_tensor)
		
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
				next_observation_tensor_combined = None
			else:
				next_observation_tensor = self.actor.to_tensor_observation(observation)
				next_observation_tensor_combined = self.actor.combine_frame_mem_and_observation(next_observation_tensor)


			# Store the transition in memory
			action_tensor = self.actor.to_tensor_action(action)
			self.memory.push(observation_tensor_combined, action_tensor, next_observation_tensor_combined, reward_tensor)

			# Move to the next observation
			observation_tensor = next_observation_tensor
			observation_tensor_combined = next_observation_tensor_combined

			# Perform one step of the optimization (on the policy network)
			self.optimize_model()

			# Soft update target net
			self.update_target_actor_from_policy_net()

			if done:
				self.episode_ended(last_step_reward=reward, steps=steps, episode_reward=episode_reward)
				break
		return reward, steps, episode_reward, episode_recording
