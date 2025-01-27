import algos.core as core
import torch
import torch.nn as nn
import math
import random
from itertools import count
import pathlib
import numpy as np
mod_name = pathlib.Path(__file__).stem

class Modifier():
	def __init__(self, algo):
		self.algo = algo
		self.algo.name += mod_name
		self.algo.saver.filename += mod_name
		self.algo.post_do_step_mod.append(self)
		self.bins_per_dim = 10
		obs_space = self.algo.env.observation_space #.shape[0]
		self.obs_high = obs_space.high
		self.obs_low = obs_space.low
		print(obs_space)
		#self.explore_reward_bins = [False]*int((self.max_position-self.min_position)/self.explore_reward_step+1)
		
	def post_do_step(self, observation2, reward, terminated, truncated, _):
		# do something with the reward value
		print(f"post_do_step reward = {reward}")
		return observation2, reward, terminated, truncated, _

