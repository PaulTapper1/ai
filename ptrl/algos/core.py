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
	def select_action(self,observation):
		# implement me
		return []
	
	def to_tensor_observation(self, observation):
		return observation
	
	def visualize(self, create_env_fn, num_episodes = 5, select_action_fn=None):
		env_visualize = create_env_fn(render_mode="human")  # Use "human" for visualization
		for i_episode in range(num_episodes):
			observation, info = env_visualize.reset()
			observation = self.to_tensor_observation(observation)
			episode_reward = 0

			for steps in count():
				env_visualize.render()  # Render the environment
				if select_action_fn == None:
					action = self.select_action(observation)
				else:
					action = select_action_fn(steps, observation)
				observation, reward, terminated, truncated, _ = env_visualize.step(action)
				episode_reward += reward

				observation = self.to_tensor_observation(observation)
				if terminated or truncated:
					print(f"visualize_model: episode {i_episode + 1} ended: steps = {steps+1}, episode_reward = {episode_reward:0.1f}, last step reward = {reward:0.1f}")
					break
		env_visualize.close()
		
	def visualize_model_from_recording(self, create_env_fn, episode_recording, num_episodes = 5):
		self.visualize(create_env_fn, num_episodes=num_episodes, 
					   select_action_fn = ( lambda steps, observation : episode_recording[steps] ) )
		
	def create_copy(self):
		return copy.deepcopy(self)

class MLPActor(ActorCore):
	def __init__(self, observation_dim, action_dim, hidden_layer_sizes=[256,256],
				 activation=nn.ReLU, output_activation=nn.Identity):
		super().__init__()
		self.mlp = MLP(observation_dim=observation_dim, action_dim=action_dim, 
						 hidden_layer_sizes=hidden_layer_sizes, activation=activation)
		self.device = get_device()
	
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
		
class MLPActorDiscreteActions(MLPActor):
	def __init__(self, create_env_fn, hidden_layer_sizes=[256,256],
				 activation=nn.ReLU, output_activation=nn.Identity):
		temp_env = create_env_fn()
		if not f"{temp_env.action_space}".startswith("Discrete"):
			print_error("Cannot create an MLPActorDiscreteActions using a continuous action space")
		action_dim = temp_env.action_space.n
		state, info = temp_env.reset()
		observation_dim = len(state)
		#print(f"MLPActorDiscreteActions: environment {temp_env.spec.id} observation_dim = {observation_dim}, action_dim = {action_dim}")
		temp_env.close()
		super().__init__(observation_dim=observation_dim, action_dim=action_dim,
						 hidden_layer_sizes=hidden_layer_sizes, activation=activation)

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
			
		fig = plt.figure(num=1)
		plt.clf()
		fig.canvas.manager.set_window_title(self.name)
		num_graphs = len(data_to_plot)
		height_ratios = [1]*num_graphs
		height_ratios[0] = 3			# make the main graph taller than the rest
		gs = gridspec.GridSpec(num_graphs, 1, height_ratios=height_ratios, hspace=0.8)
		fontsize = 8
		linewidth = 1

		for subplot, data_name in enumerate(data_to_plot):
			ax = fig.add_subplot(gs[subplot])
			if subplot == len(data_to_plot)-1:
				ax.set_xlabel('Episode', fontsize=fontsize)
				ax.tick_params(axis='x', which='major', labelsize=8)  
			else:
				ax.set_xticks([])
			ax.tick_params(axis='y', which='major', labelsize=8)  
			ax.set_title(data_name, fontsize=fontsize, loc="left")
			ax.plot(self.data[data_name], linewidth=linewidth)

			# Draw a smoothed graph
			if subplot == 0 and smooth > 0:
				if len(self.data[data_name]) >= smooth:
					window = np.ones(int(smooth))/float(smooth)
					smoothed = np.convolve(self.data[data_name], window, 'valid')
					ax.plot(np.arange(smooth//2, smooth//2+len(smoothed)),smoothed, linewidth=linewidth)
		
		plt.pause(0.01)  # pause a bit so that plots are updated
		plt.show(block=block)
		

#####################################################################################
# AlgoBase
import inspect

class AlgoBase:
	def __init__(self, name, create_env_fn, settings):
		self.settings = settings	# should be a Dict
		
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
		self.EPS_DECAY = 50   # based on episodes
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
		self.steps_done = 0
		self.average_duration = 0
		self.device = get_device()
		
	def get_save_name(self):
		save_name = self.name+"_"+self.env_name
		for hidden_layer_size in self.settings["hidden_layer_sizes"]:
			save_name += "_"+str(hidden_layer_size)
		if "experiment" in self.settings :
			save_name += "_ex"+str(self.settings["experiment"])
		return save_name

	def save(self):
		print(f"{inspect.currentframe().f_code.co_name} overload in child classes")

	def load(self):
		print(f"{inspect.currentframe().f_code.co_name} overload in child classes")
		
	def post_load_fixup(self):
		self.steps_done = self.logger.get_latest_value("steps_done")
		
	def load_if_save_exists(self):
		if self.saver.save_exists():
			self.load()
		
	def visualize(self, num_episodes = 5):
		self.actor.visualize(self.create_env_fn, num_episodes)
		

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
		
	def plot(self, block=False):
		plt.figure(num=self.plot_figure_num)
		plt.clf()
		plt.title("Experiment: "+self.saver.filename)
		plt.grid(axis='x')  # Add vertical grid lines
		plt.xticks(np.arange(len(self.completed_experiments.items())))
		for num, (key, data) in enumerate(self.completed_experiments.items()):
			data.sort()
			x_coords = np.arange(len(data))*(0.5/len(data)) + float(num)
			plt.plot( x_coords, data, "ro")
	
		plt.pause(0.01)  # pause a bit so that plots are updated
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
	
	

