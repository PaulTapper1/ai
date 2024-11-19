import core
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

class Algo(core.AlgoBase):
	def __init__(self, create_env_fn, settings):
		super().__init__(create_env_fn=create_env_fn, settings=settings)
		self.settings = settings	# should be a Dict
		self.actor = core.MLPActorDiscreteActions(self.create_env_fn, hidden_layer_sizes=self.settings["hidden_layer_sizes"])
		self.optimizer = optim.AdamW(self.actor.mlp.parameters(), lr=self.LR, amsgrad=True)
		self.load_if_save_exists()
		self.target_net = core.MLPActorDiscreteActions(self.create_env_fn, hidden_layer_sizes=self.settings["hidden_layer_sizes"])
		self.target_net.mlp.load_state_dict(self.actor.mlp.state_dict())

	def save(self):
		self.saver.add_data_to_save( "mlp",	self.actor )
		self.saver.add_data_to_save( "mem",	self.memory )
		self.saver.add_data_to_save( "json", self.settings, method="json" )
		self.saver.save()
	
	def load(self):
		self.saver.load_data_into( "mlp",	self.actor, is_net=True )
		self.saver.load_data_into( "mem",	self.memory )
		self.saver.load_data_into( "json", self.settings, method="json" )
		self.saver.save()
	
	def get_epsilon(self):
		decay = self.episodes_done		 # epsilon based on episodes
		return self.EPS_END + (self.EPS_START - self.EPS_END) * \
			math.exp(-1. * decay / self.EPS_DECAY)

	def select_action(self, observation):
		eps_threshold = self.get_epsilon()
		self.steps_done += 1
		if random.random() > eps_threshold:
			action = self.actor.select_action(observation)
			#print(f"select_action: policy_action = {action}")
			return action
		else:
			random_action = self.env.env.action_space.sample()
			#print(f"select_action: random action = {random_action}")
			return random_action

	def visualize(self, num_episodes = 5):
		self.actor.visualize(self.create_env_fn, num_episodes)

#####################################################################################
# for testing
if __name__ == '__main__':
	import gymnasium as gym
	
	print("Running code tests...")
	
	print("\nTesting Algo...")
	settings = { "hidden_layer_sizes" : [128,64,32] }

	# algo = Algo( create_env_fn = (lambda **kwargs : gym.make("LunarLander-v3",*kwargs)), settings=settings )	
	# # https://gymnasium.farama.org/environments/box2d/lunar_lander/
	# algo.visualize()

	def create_env_fn_LunarLander(render_mode=None):
		return gym.make("LunarLander-v3",render_mode=render_mode)		# https://gymnasium.farama.org/environments/box2d/lunar_lander/
	algo = Algo( create_env_fn_LunarLander, settings=settings )	
	algo.visualize()
