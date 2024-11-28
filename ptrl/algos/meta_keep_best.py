import algos.core as core
import torch
import torch.nn as nn
import math
import random
from itertools import count
import pathlib
import numpy as np
meta_algo_name = pathlib.Path(__file__).stem

class MetaAlgo(core.AlgoBase):
	def __init__(self, create_env_fn, settings=[], algo=None):
		super().__init__(name=meta_algo_name+"_"+algo.name, create_env_fn=create_env_fn, settings=settings)
		self.algo = algo
		self.algo.name = self.name
		self.algo.saver.filename = self.saver.filename
		self.algo.save_handler = self
		self.algo.logger.name = self.logger.name
		self.algo.episode_ended_handler = self	# use to change standard behaviour (eg- for a meta-algorithm)
		self.algo.data_to_plot=["episode_reward",["most_recent_actor_score","best_actor_score"],"episodes_before_revert_to_best_cursor","last_step_reward","episode_durations","epsilon"]

		self.best_actor = self.algo.actor.create_copy()
		self.best_actor_score = -500				# TODO - get a better way of finding a starting value
		self.most_recent_actor_score = self.best_actor_score
		self.episodes_before_revert_to_best = 100
		self.episodes_before_revert_to_best_cursor = self.episodes_before_revert_to_best
		self.cooldown_after_tests = 20
		self.epsilon_multiplier_on_revert_to_best = 4
		self.cooldown_after_tests_cursor = self.cooldown_after_tests
		self.num_test_episodes = 5
		self.test_seed = np.random.randint(1000)
		self.load_if_save_exists()

	def save(self):
		self.add_data_to_save()
		self.algo.saver.save()

	def add_data_to_save(self):
		self.algo.add_data_to_save()
		self.algo.saver.add_data_to_save( "best_actor",		self.best_actor.mlp, 	is_net=True )
	
	def load(self):
		self.algo.load()
		self.algo.saver.load_data_into( "best_actor",		self.best_actor.mlp, 	is_net=True )
		self.best_actor_score = self.algo.logger.get_latest_value("best_actor_score")
		self.most_recent_actor_score = self.algo.logger.get_latest_value("most_recent_actor_score")
		self.test_seed = self.algo.logger.get_latest_value("test_seed")
		self.episodes_before_revert_to_best_cursor = self.algo.logger.get_latest_value("episodes_before_revert_to_best_cursor")
		self.cooldown_after_tests_cursor = self.algo.logger.get_latest_value("cooldown_after_tests_cursor")

		# # Hack to fix up some data in save
		# data_to_fix = self.algo.logger.data["best_actor_score"]
		# for i in range(len(data_to_fix)):
			# if data_to_fix[i] < -500:
				# data_to_fix[i] = -500
		# data_to_fix = self.algo.logger.data["most_recent_actor_score"]
		# for i in range(len(data_to_fix)):
			# if data_to_fix[i] < -500:
				# data_to_fix[i] = -500

	def visualize(self, **kwargs):
		temp_actor = self.algo.actor
		self.algo.actor = self.best_actor
		self.algo.visualize(**kwargs)
		self.algo.actor = temp_actor

	def loop_episodes(self, **kwargs):
		self.algo.loop_episodes(**kwargs)

	def get_score_for_actor(self, actor, test_name=""):
		results, average = actor.test(create_env_fn=self.algo.create_env_fn,
									  num_test_episodes=self.num_test_episodes, 
									  seed_offset=self.test_seed,
									  test_name=test_name)
		return average

	def episode_ended(self, last_step_reward, steps, episode_reward):
		# check for and update the best actor
		if self.cooldown_after_tests_cursor > 0:
			self.cooldown_after_tests_cursor -= 1
		else:
			if episode_reward > self.best_actor_score:
				self.cooldown_after_tests_cursor = self.cooldown_after_tests
				self.test_seed += self.num_test_episodes
				self.best_actor_score = self.get_score_for_actor(actor=self.best_actor, test_name="best actor ")
				self.most_recent_actor_score = self.get_score_for_actor(actor=self.algo.actor, test_name="current actor ")
				if self.most_recent_actor_score > self.best_actor_score:
					self.best_actor_score = self.most_recent_actor_score
					self.best_actor = self.algo.actor.create_copy()
					self.episodes_before_revert_to_best_cursor = self.episodes_before_revert_to_best
					print(f"Updated best actor (score = {self.best_actor_score:0.1f})")
				else:
					print(f"Keeping best actor (score = {self.best_actor_score:0.1f})")
		
		if self.episodes_before_revert_to_best_cursor > 0:
			self.episodes_before_revert_to_best_cursor -= 1
		else:
			self.episodes_before_revert_to_best_cursor = self.episodes_before_revert_to_best
			self.algo.actor = self.best_actor.create_copy()
			self.most_recent_actor_score = self.best_actor_score
			self.algo.epsilon = min( self.algo.EPS_START, self.algo.epsilon * self.epsilon_multiplier_on_revert_to_best)
			print(f"Reverting to best actor (score = {self.best_actor_score:0.1f}) and increasing epsilon")
									
		self.algo.logger.set_frame_value("most_recent_actor_score",	self.most_recent_actor_score)
		self.algo.logger.set_frame_value("best_actor_score",		self.best_actor_score)
		self.algo.logger.set_frame_value("test_seed",				self.test_seed)
		self.algo.logger.set_frame_value("episodes_before_revert_to_best_cursor",	self.episodes_before_revert_to_best_cursor)
		self.algo.logger.set_frame_value("cooldown_after_tests_cursor",	self.cooldown_after_tests_cursor)
		self.algo.episode_ended(last_step_reward, steps, episode_reward)		# do all the standard stuff at the end of an episode
		# saving will be triggered by algo

	def test_actor(self, **kwargs):
		results, average = self.best_actor.test(create_env_fn=self.algo.create_env_fn, **kwargs)
		return results
