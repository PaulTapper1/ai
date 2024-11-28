"""
Code based in places on spinningup https://spinningup.openai.com/en/latest/index.html

You may need to
	pip3 install "gymnasium[classic-control]"
	pip3 install gymnasium

to see a list of all available gyms use...
	import gymnasium as gym
	gym.pprint_registry()
	
see also https://github.com/openai/gym/wiki/Leaderboard
"""

#####################################################################################
# Console output
def print_warning(message):
	print('\033[93mWarning: '+message+'\033[0m')	# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
def print_error(message):
	print('\033[91mError: '+message+'\033[0m')	# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
	exit()

#####################################################################################
# Torch interface classes
import torch
import torch.nn as nn
import torch.optim as optim

device = False

def get_device():	# Get device to run torch on (CPU or GPU)
	global device
	if device == False:
		device = torch.device(
			"cuda" if torch.cuda.is_available() else
			"mps" if torch.backends.mps.is_available() else
			"cpu"
			)
		print(f"Torch device = {device}")
	return device
get_device()

# Multi-Layer Perceptron class
class MLP(nn.Module):
	def __init__(self, observation_dim, action_dim, hidden_layer_sizes=[256,256],
				 activation=nn.ReLU, output_activation=nn.Identity):
		super().__init__()
		self.observation_dim = observation_dim
		self.action_dim = action_dim
		self.hidden_layer_sizes = hidden_layer_sizes
		layer_sizes = [self.observation_dim] + self.hidden_layer_sizes + [self.action_dim]
		layers = []
		for j in range(len(layer_sizes)-1):
			act = activation if j < len(layer_sizes)-2 else output_activation
			layers += [nn.Linear(layer_sizes[j], layer_sizes[j+1]), act()]
		self.linear_relu_stack = nn.Sequential(*layers)
		self = self.to(get_device())
		
	# Called with either one element to determine next action, or a batch
	# during optimization.
	def forward(self, x):
		return self.linear_relu_stack(x)
	
	def get_brief_str(self):
		return str(self.__class__.__name__)+"_"+"_".join( str(size) for size in self.hidden_layer_sizes)

#####################################################################################
# Actor classes
from itertools import count
import copy

class ActorCore:
	def reset(self):
		pass	# empty function - but can be overloaded in child classes
		
	def select_action(self,observation):
		# implement me
		return []
	
	def to_tensor_observation(self, observation):
		return observation
	
	def visualize(self, create_env_fn, num_episodes=5):
		for i_episode in range(num_episodes):
			reward, steps, episode_reward = self.do_episode(create_env_fn=create_env_fn, visualize=True)
			print(f"visualize: episode {i_episode + 1} ended: steps = {steps+1}, episode_reward = {episode_reward:0.1f}, last step reward = {reward:0.1f}")
	
	def test(self, create_env_fn, num_test_episodes=20, seed_offset = 0, visualize=False, test_name=""):
		#print(f"Running {num_test_episodes} test episodes (seed_offset = {seed_offset})")
		results = []
		for test_number in range(num_test_episodes):
			seed = seed_offset+test_number
			last_step_reward, steps, episode_reward = self.do_episode(create_env_fn=create_env_fn, 
									  seed=seed, visualize=visualize)
			print(f"test {test_name}episode {test_number} (seed {seed}) ended: steps = {steps+1}, episode_reward = {episode_reward:0.1f}, last step reward = {last_step_reward:0.1f}")
			results.append(episode_reward)
		average = np.mean(np.array(results))
		print(f"After {num_test_episodes} tests, got average epsiode score = {average:0.1f}")
		return results, average

	# run the actor on policy for one episode (optionally visualizing it) and return the results
	def do_episode(self, create_env_fn, visualize=False, seed=None):
		if visualize:
			env = create_env_fn(render_mode="human")  # Use "human" for visualization
		else:
			env = create_env_fn()
		observation, info = env.reset(seed=seed)
		self.reset()
		observation = self.to_tensor_observation(observation)
		episode_reward = 0
		for steps in count():
			# if visualize:
				# env.render()  # Render the environment
			action = self.select_action(observation)
			observation, reward, terminated, truncated, _ = env.step(action)
			episode_reward += reward
			observation = self.to_tensor_observation(observation)
			if terminated or truncated:
				break
		env.close()
		return reward, steps, episode_reward
		
	def create_copy(self):
		return copy.deepcopy(self)

class MLPActor(ActorCore):
	def __init__(self, observation_dim, action_dim, hidden_layer_sizes=[256,256],
				 activation=nn.ReLU, output_activation=nn.Identity, learning_rate=1e-4, **kwargs):
		super().__init__()
		self.mlp = MLP(observation_dim=observation_dim, action_dim=action_dim, 
						 hidden_layer_sizes=hidden_layer_sizes, activation=activation)
		self.device = get_device()
		self.learning_rate = learning_rate
		self.optimizer = optim.AdamW(self.mlp.parameters(), lr=self.learning_rate, amsgrad=True)
	
	def select_action(self,observation):
		with torch.no_grad():
			return self.mlp.forward(observation)

	def to_tensor_observation(self, observation):
		return torch.tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)
		
	def to_tensor_action(self, action):
		return torch.tensor([[action]], device=self.device, dtype=torch.long)
		
	def to_tensor_reward(self, reward):
		return torch.tensor([reward], device=self.device)
		
	def create_copy(self):
		copied_object = super().create_copy()
		copied_object.mlp.load_state_dict(self.mlp.state_dict())
		return copied_object
		
	def optimize(self, loss, clip_grad_value=None):
		self.optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		if clip_grad_value is not None:
			torch.nn.utils.clip_grad_value_(self.mlp.parameters(), clip_grad_value)
		self.optimizer.step()
		
		
class MLPActorDiscreteActions(MLPActor):
	def __init__(self, create_env_fn, hidden_layer_sizes=[256,256],
				 activation=nn.ReLU, output_activation=nn.Identity, **kwargs):
		temp_env = create_env_fn()
		if not f"{temp_env.action_space}".startswith("Discrete"):
			print_error("Cannot create an MLPActorDiscreteActions using a continuous action space")
		action_dim = temp_env.action_space.n
		state, info = temp_env.reset()
		observation_dim = len(state)
		#print(f"MLPActorDiscreteActions: environment {temp_env.spec.id} observation_dim = {observation_dim}, action_dim = {action_dim}")
		temp_env.close()
		super().__init__(observation_dim=observation_dim, action_dim=action_dim,
						 hidden_layer_sizes=hidden_layer_sizes, activation=activation, **kwargs)

	def select_action(self,observation):
		raw_action = super().select_action(observation)
		# t.max(1) will return the largest column value of each row.
		# second column on max result is index of where max element was
		# found, so we pick action with the larger expected reward.
		action = raw_action.max(1).indices.view(1, 1).item()
		return action
		
#####################################################################################
# Saving and Loading
import os
import uuid
import json
import glob

class Saver:
	def __init__(self, filename, folder="data"):
		if not os.getcwd().endswith(folder):	# move into data subfolder
			os.chdir(folder)
		self.filename = filename
		self.folder = folder
		self.data_to_save = {}
	
	def add_data_to_save(self, extension, data, is_net=False):
		data_descriptor = { "data" : data, "is_net" : is_net }
		self.data_to_save [extension] = data_descriptor
	
	def save(self):
		temp_filename = "temp_"+str(uuid.uuid4())
		for extension, data_descriptor in self.data_to_save.items():
			data = data_descriptor["data"]
			is_net = data_descriptor["is_net"]
			if is_net:
				torch.save(data.state_dict(), temp_filename+"."+extension)
			else:
				torch.save(data.to_saveable(), temp_filename+"."+extension)
		for extension in self.data_to_save:
			if os.path.isfile(self.filename+"."+extension):
				os.remove(self.filename+"."+extension)
		for extension in self.data_to_save:
			os.rename(temp_filename+"."+extension, self.filename+"."+extension)
		self.data_to_save = {}
		print(f"Saved {self.filename}")
	
	def save_exists(self):
		return (len(glob.glob(self.filename+".*")) > 0)

	def load_data_into(self, extension, obj, is_net=False):
		loaded_data = torch.load(self.filename+"."+extension, weights_only=False)
		if is_net:
			obj.load_state_dict(loaded_data)
			obj.eval()  					# Set the net to evaluation mode
		else:
			obj.from_saveable( loaded_data )

#####################################################################################
# AlgoMemory
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class AlgoMemory(object):

	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
	
	def to_saveable(self):
		return self.memory
		
	def from_saveable(self, saveable):
		self.memory = saveable

#####################################################################################
# Logging and Graphing
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

class Logger():
	def __init__(self, name=""):
		self.data = {}
		self.num_frames = 0
		self.name = name
		
	def set_frame_value(self, key, value):
		if not key in self.data :
			if self.num_frames == 0 :
				self.data[key] = []
			else:
				print_warning(f"Adding unrecognised key {key} at frame {self.num_frames} to Logger")
				self.data[key] = [0]*self.num_frames
		self.data[key].append(value)
	
	def get_latest_value(self, key):
		if not key in self.data :
			#print_warning(f"Could not find key {key} in {self}")
			return 0
		data_list = self.data[key]
		if len(data_list) > 0 :
			return data_list[len(data_list)-1]
		else:
			return 0
	
	def next_frame(self):
		self.num_frames += 1
		for key in self.data.keys():
			if len(self.data[key]) < self.num_frames:
				print_warning(f"Missing {key} value for frame {self.num_frames-1}")
				
	def to_saveable(self):
		return self.data
		
	def from_saveable(self, saveable):
		self.data = saveable
		self.num_frames = 0
		if len(self.data.keys()) > 0 :
			self.num_frames = len(self.data[next(iter(self.data))])
		
	def __str__(self):
		ret = ""
		ret += f"Logger: Num_frames = {self.num_frames}\n"
		for key in self.data.keys():
			ret += f"{key} : {self.data[key]}\n"
		return ret
	
	def plot(self, data_to_plot=None, block=False, smooth=30):
		if data_to_plot == None:
			data_to_plot = self.data.keys()
			
		fig = plt.figure(num=1) #, figsize=(12,8))
		plt.clf()
		fig.canvas.manager.set_window_title(self.name)
		num_graphs = len(data_to_plot)
		height_ratios = [1]*num_graphs
		height_ratios[0] = 3			# make the main graph taller than the rest
		height_ratios[1] = 3			# make the main graph taller than the rest
		gs = gridspec.GridSpec(num_graphs, 1, height_ratios=height_ratios, hspace=0.8)
		fontsize = 8
		linewidth = 1

		for subplot, data_names in enumerate(data_to_plot):
			ax = fig.add_subplot(gs[subplot])
			if subplot == len(data_to_plot)-1:
				ax.set_xlabel('Episode', fontsize=fontsize)
				ax.tick_params(axis='x', which='major', labelsize=8)  
			else:
				ax.set_xticks([])
			ax.tick_params(axis='y', which='major', labelsize=8)
			
			if type(data_names) == str:
				data_names = [data_names]
			chart_title = ""
			for data_name in data_names:
				if chart_title != "":
					chart_title += " / "
				chart_title += data_name
				if data_name in self.data:
					this_data = self.data[data_name]
					chart_title += f" ({this_data[len(this_data)-1]:0.1f})"
					ax.plot(this_data, linewidth=linewidth)

					# Draw a smoothed graph
					if subplot == 0 and smooth > 0:
						if len(this_data) >= smooth:
							window = np.ones(int(smooth))/float(smooth)
							smoothed = np.convolve(this_data, window, 'valid')
							ax.plot(np.arange(smooth//2, smooth//2+len(smoothed)),smoothed, linewidth=linewidth)
			ax.set_title(chart_title, fontsize=fontsize, loc="left")
		
		plt.pause(0.1)  # pause a bit so that plots are updated
		plt.show(block=block)
		

#####################################################################################
# AlgoBase
import math

def generic_get_save_name(algo_name, env_name, settings):
	save_name = algo_name+"_"+env_name
	if "hidden_layer_sizes" in settings:
		for hidden_layer_size in settings["hidden_layer_sizes"]:
			save_name += "_"+str(hidden_layer_size)
	return save_name

class AlgoBase:
	def __init__(self, name, create_env_fn, settings):
		self.settings = Saveable(settings)
		
		# TODO - make these settings below part of self.settings
		
		# BATCH_SIZE is the number of transitions sampled from the replay buffer
		# GAMMA is the discount factor as mentioned in the previous section
		# EPS_START is the starting value of epsilon
		# EPS_END is the final value of epsilon
		# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
		# TAU is the update rate of the target network
		# LR is the learning rate of the ``AdamW`` optimizer
		self.BATCH_SIZE = 64
		self.GAMMA = 0.99
		self.EPS_START = 0.9
		self.EPS_END = 0.05
		#self.EPS_DECAY = 5000 #1000   # based on steps
		self.EPS_HALF_LIFE = 20    # based on episodes
		self.TAU = 0.005
		self.LR = 1e-4
		self.MEM_SIZE = 10000
		
		self.name = name
		self.create_env_fn = create_env_fn
		self.env = create_env_fn()
		self.env_name = self.env.spec.name
		save_name = self.get_save_name()
		self.logger = Logger(save_name)
		self.memory = AlgoMemory(self.MEM_SIZE)
		self.saver = Saver(save_name)
		self.save_handler = None
		self.save_every_frames = 5
		self.steps_done = 0
		self.device = get_device()
		self.data_to_plot = ["episode_reward","last_step_reward","episode_durations","memory_size"]
		self.episode_ended_handler = None	# use to change standard behaviour (eg- for a meta-algorithm)
		self.epsilon = self.EPS_START
		self.epsilon_decay = math.exp(-math.log(2.) / self.EPS_HALF_LIFE)
		
	def get_save_name(self):
		save_name = generic_get_save_name( self.name, self.env_name, self.settings )
		if "experiment" in self.settings :
			save_name += "_ex"+str(self.settings["experiment"])
		return save_name

	def save(self):
		if self.save_handler is not None:
			self.save_handler.save()
		else:
			self.add_data_to_save()
			self.saver.save()		

	def add_data_to_save(self):
		self.saver.add_data_to_save( "memory",			self.memory )
		self.saver.add_data_to_save( "settings", 		self.settings )
		self.saver.add_data_to_save( "logger",			self.logger )

	def load(self):
		self.saver.load_data_into( "memory",			self.memory )
		self.saver.load_data_into( "settings", 			self.settings )
		self.saver.load_data_into( "logger", 			self.logger )
		self.steps_done 	= self.logger.get_latest_value("steps_done")
		self.epsilon 		= self.logger.get_latest_value("epsilon")
				
	def load_if_save_exists(self):
		if self.saver.save_exists():
			self.load()
		
	def visualize(self, num_episodes=5):
		self.actor.visualize(create_env_fn=self.create_env_fn, num_episodes=num_episodes)

	def loop_episodes(self, num_episodes, visualize_every=0, show_graph=True):
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

	def episode_ended_outer(self, last_step_reward, steps, episode_reward):
		print(f"episode_ended {self.logger.get_latest_value('episodes') + 1}: steps = {steps+1}, episode_reward = {episode_reward:0.1f}, last step reward = {last_step_reward:0.1f}") 
		if self.episode_ended_handler is not None:
			return self.episode_ended_handler.episode_ended(last_step_reward, steps, episode_reward)
		return self.episode_ended(last_step_reward, steps, episode_reward)

	def episode_ended(self, last_step_reward, steps, episode_reward):
		this_episode = self.logger.get_latest_value("episodes") + 1
		self.decay_epsilon()
		self.logger.set_frame_value("episodes",						this_episode)
		self.logger.set_frame_value("steps_done",					self.steps_done)
		self.logger.set_frame_value("memory_size",					len(self.memory))
		self.logger.set_frame_value("episode_durations",			steps + 1)
		self.logger.set_frame_value("episode_reward",				episode_reward)
		self.logger.set_frame_value("last_step_reward",				last_step_reward)
		self.logger.set_frame_value("epsilon",						self.epsilon)
		self.logger.next_frame()
		if self.save_every_frames > 0:
			if self.logger.get_latest_value("episodes")%self.save_every_frames == 0:
				self.save()
			
	def test_actor(self, **kwargs):
		return self.actor.test(create_env_fn=self.create_env_fn, **kwargs)

	def show_graph(self):
		self.logger.plot(self.data_to_plot)
		
	def decay_epsilon(self):
		self.epsilon = (self.epsilon-self.EPS_END)*self.epsilon_decay + self.EPS_END

#####################################################################################
# Saveable
class Saveable(dict):
	def to_saveable(self):
		return self
		
	def from_saveable(self, saveable):
		self = saveable

#####################################################################################
# Experiment
plot_figure_num = 2

class Experiment(Saveable):
	def __init__(self, name, experiment_options):
		self.experiment_options = experiment_options
		self.iterator_cursor = [0]*len(self.experiment_options)
		self.experiment = self._get_experiment_from_cursor()
		self.saver = Saver(name)
		self.completed_experiments = {}
		if self.saver.save_exists():
			self.saver.load_data_into("experiment", self)
		global plot_figure_num
		self.plot_figure_num = plot_figure_num
		plot_figure_num += 1

	def to_saveable(self):
		return self.completed_experiments

	def from_saveable(self, saveable):
		self.completed_experiments = saveable

	def _get_experiment_from_cursor(self):
		ret = []
		for layer in range(len(self.experiment_options)):
			ret.append( self.experiment_options[layer][self.iterator_cursor[layer]] )
		return ret

	def iterate_inner(self):	# returns True if it is still iterating, and False when its finished
		if self.iterator_cursor[0] == -1:	# returned last iteration previous time this was called, so now time to terminate loop
			return False
		cursor_layer = 0
		while True:
			if cursor_layer == len(self.experiment_options):
				self.iterator_cursor[0] = -1	# will cause a loop termination next time
				break
			self.iterator_cursor[cursor_layer] += 1
			if self.iterator_cursor[cursor_layer] < len(self.experiment_options[cursor_layer]):
				break
			self.iterator_cursor[cursor_layer] = 0
			cursor_layer += 1
		return True

	def iterate(self):	# returns True if it is still iterating, and False when its finished
		self.experiment = self._get_experiment_from_cursor()
		while True:
			ret = self.iterate_inner()
			if not self.get_experiment_str() in self.completed_experiments:
				break
			if ret == False:
				break
			self.experiment = self._get_experiment_from_cursor()
		return ret
	
	def get_experiment_str(self):
		return str(self.experiment)

	def experiment_completed(self, results):
		self.completed_experiments [self.get_experiment_str()] = results
		self.saver.add_data_to_save("experiment", self)
		self.saver.save()
		
	def plot(self, block=False, save_image=False):
		fontsize = 8
		plt.figure(num=self.plot_figure_num)
		plt.clf()
		plt.title("Experiment: "+self.saver.filename, fontsize=fontsize)
		plt.grid(axis='x')  # Add vertical grid lines
		plt.grid(axis='y')  # Add vertical grid lines
		plt.xticks(np.arange(len(self.completed_experiments.items())))
		plt.yticks([-200,-100,0,100,200,300])
		plt.tick_params(axis='x', which='major', labelsize=8)  
		plt.tick_params(axis='y', which='major', labelsize=8)  
		for num, (key, data) in enumerate(self.completed_experiments.items()):
			data.sort()
			x_coords = np.arange(len(data))*(0.5/len(data)) + float(num)
			all_points = list(zip(x_coords,data))
			lose_points = [p for p in all_points if p[1] < 200]
			win_points = [p for p in all_points if p[1] >= 200]
			plt.plot( [p[0] for p in lose_points], [p[1] for p in lose_points], "ro", markersize=5)
			plt.plot( [p[0] for p in win_points], [p[1] for p in win_points], "go", markersize=5)
	
		plt.pause(0.01)  # pause a bit so that plots are updated
		if save_image:
			image_filename = self.saver.filename+".png"
			plt.savefig(image_filename)
			print(f"Saved image {image_filename}")
		plt.show(block=block)
		

#####################################################################################
# for testing
if __name__ == '__main__':
	import time
	import gymnasium as gym
	
	print("Running code tests...")
	
	print("\nTesting MLP...")
	mlp = MLP(6,3,[128,64,32])
	print(f"{mlp}")
	print(f"Brief string = {mlp.get_brief_str()}")
	
	print("\nTesting Saver...")
	saver = Saver("test_save")
	obj1 = [1,2,3,4,5]
	saver.add_data_to_save("obj1", obj1, "json")
	obj2 = { "a" : 2, "b" : 4, "c" : 6 }
	saver.add_data_to_save("obj2", obj2, "json")
	saver.save()
	if saver.exists():
		saver.load_data_into("obj1", obj1, "json")
		saver.load_data_into("obj2", obj2, "json")
	
	print("\nTesting MLPActor...")
	actor = MLPActor()
	
	print("\nTesting Logger...")
	logger = Logger()
	logger.add("Test1",1)
	logger.add("Test2",2)
	logger.next_frame()
	logger.add("Test3",3)
	logger.add("Test1",4)
	logger.next_frame()
	print(logger)
	logger.plot()
	time.sleep(1)
	
	

