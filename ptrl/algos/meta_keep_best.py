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
		self.algo.data_to_plot=["episode_reward",["most_recent_actor_score","best_actor_score"],"last_step_reward","episode_durations"]

		self.best_actor = self.algo.actor.create_copy()
		self.best_actor_score = -1000				# TODO - get a better way of finding a starting value
		self.most_recent_actor_score = -1000			# TODO - get a better way of finding a starting value
		self.update_best_actor_every_frames = 10
		self.revert_to_best_actor_every_actor_updates = 4
		self.num_test_episodes = 5
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

	def visualize(self, **kwargs):
		temp_actor = self.algo.actor
		self.algo.actor = self.best_actor
		self.algo.visualize(**kwargs)
		self.algo.actor = temp_actor

	def loop_episodes(self, **kwargs):
		self.algo.loop_episodes(**kwargs)

	def episode_ended(self, last_step_reward, steps, episode_reward):
		# log out any algorithm specific data you want to track
		
		# check for and update the best actor so far
		if self.algo.logger.get_latest_value("episodes")%self.update_best_actor_every_frames == self.update_best_actor_every_frames-1:
			results = self.algo.test_actor(num_test_episodes=self.num_test_episodes)
			self.most_recent_actor_score = np.mean(np.array(results))
			if self.most_recent_actor_score > self.best_actor_score:
				self.best_actor_score = self.most_recent_actor_score
				self.best_actor = self.algo.actor
				print(f"Updated best actor (score = {self.best_actor_score:0.1f})")
				
		# if best actor is better than most recently tested actor, then revert back to the best one and continue from there again
		revert_to_best_actor_every_frames = self.revert_to_best_actor_every_actor_updates * self.update_best_actor_every_frames
		if self.algo.logger.get_latest_value("episodes")%revert_to_best_actor_every_frames == revert_to_best_actor_every_frames-1:
			if self.most_recent_actor_score < self.best_actor_score:
				self.algo.actor = self.best_actor
				self.most_recent_actor_score = self.best_actor_score
				print(f"Reverting best actor (score = {self.best_actor_score:0.1f})")
				
		self.algo.logger.set_frame_value("most_recent_actor_score",	self.most_recent_actor_score)
		self.algo.logger.set_frame_value("best_actor_score",	self.best_actor_score)
		self.algo.episode_ended(last_step_reward, steps, episode_reward)		# do all the standard stuff at the end of an episode
		# saving will be triggered by algo

	def test_actor(self, **kwargs):
		temp_actor = self.algo.actor
		self.algo.actor = self.best_actor
		results = self.algo.test_actor(**kwargs)
		self.algo.actor = temp_actor
		return results
