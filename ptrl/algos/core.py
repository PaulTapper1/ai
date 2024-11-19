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

class ActorCore:
	def select_action(self,observation):
		# implement me
		return []
	
	def prep_observation(self, observation):
		return observation
	
	def visualize(self,create_env_fn,num_episodes = 5):
		env_visualize = create_env_fn(render_mode="human")  # Use "human" for visualization
		for i_episode in range(num_episodes):
			observation, info = env_visualize.reset()
			observation = self.prep_observation(observation)
			reward_total = 0

			for steps in count():
				env_visualize.render()  # Render the environment
				action = self.select_action(observation)
				observation, reward, terminated, truncated, _ = env_visualize.step(action)
				reward_total += reward

				observation = self.prep_observation(observation)
				if terminated or truncated:
					print(f"visualize_model: episode {i_episode + 1} ended: steps = {steps+1}, reward_total = {reward_total:0.1f}")
					break
		env_visualize.close()

class MLPActor(ActorCore):
	def __init__(self, observation_dim, action_dim, hidden_layer_sizes=[256,256],
				 activation=nn.ReLU, output_activation=nn.Identity):
		super().__init__()
		self.mlp = MLP(observation_dim=observation_dim, action_dim=action_dim, 
						 hidden_layer_sizes=hidden_layer_sizes, activation=activation)
	
	def select_action(self,observation):
		with torch.no_grad():
			return self.mlp.forward(observation)

	def prep_observation(self, observation):
		return torch.tensor(observation, dtype=torch.float32, device=get_device()).unsqueeze(0)
		
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
		self.save_method = {}
	
	def add_data_to_save(self, extension, data, method="torch"):	# options for method are "torch" or "json"
		self.data_to_save [extension] = data
		self.save_method [extension] = method
	
	def save(self):
		temp_filename = "temp_"+str(uuid.uuid4())
		for extension, data in self.data_to_save.items():
			if self.save_method [extension] == "torch":
				torch.save(data, temp_filename+"."+extension)
			elif self.save_method [extension] == "json":
				with open(temp_filename+"."+extension, 'w') as f:
					json.dump(data, f, indent=2)
			else:
				print_warning(f"Unrecognised save method {self.save_method [extension]} for extension {extension}")
		for extension in self.data_to_save:
			if os.path.isfile(self.filename+"."+extension):
				os.remove(self.filename+"."+extension)
		for extension in self.data_to_save:
			os.rename(temp_filename+"."+extension, self.filename+"."+extension)
		print(f"Saved {self.filename}")
	
	def save_exists(self):
		return (len(glob.glob(self.filename+".*")) > 0)

	def load_data_into(self, extension, obj, method="torch", is_net=True):	# options for method are "torch" or "json"
		if method == "torch":
			loaded_data = torch.load(self.filename+"."+extension,weights_only=False)
			if is_net:
				obj.load_state_dict(loaded_data)
				obj.eval()  					# Set the net to evaluation mode
			else:
				obj = loaded_data
		elif method == "json":
			with open(self.filename+"."+extension, 'r') as f:
				obj = json.load(f)
		else:
			print_warning(f"Unrecognised save method {method} for extension {extension}")		

#####################################################################################
# AlgoMemory
from collections import namedtuple, deque

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

#####################################################################################
# Logging and Graphing
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy

class Logger():
	def __init__(self):
		self.data = {}
		self.num_frames = 0
		
	def add(self, key, value):
		if not key in self.data:
			if self.num_frames == 0:
				self.data[key] = []
			else:
				print_warning(f"Adding unrecognised key {key} at frame {self.num_frames} to Logger")
				self.data[key] = [None]*self.num_frames
		self.data[key].append(value)
	
	def next_frame(self):
		self.num_frames += 1
		for key in self.data.keys():
			if len(self.data[key]) < self.num_frames:
				print_warning(f"Missing {key} value for frame {self.num_frames-1}")
		
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
		num_graphs = len(data_to_plot)
		height_ratios = [1]*num_graphs
		height_ratios[0] = 3			# make the main graph taller than the rest
		gs = gridspec.GridSpec(num_graphs, 1, height_ratios=height_ratios, hspace=0.5)
		fontsize = 9
		linewidth = 1

		for subplot, data_name in enumerate(data_to_plot):
			ax = fig.add_subplot(gs[subplot])  
			ax.set_xticks([])
			ax.set_title(data_name, fontsize=fontsize, loc="left")
			ax.plot(self.data[data_name], linewidth=linewidth)

			# Draw a smoothed graph
			if subplot == 0 and smooth > 0:
				if len(self.data[data_name]) >= smooth:
					window= numpy.ones(int(window_size))/float(window_size)
					smoothed = numpy.convolve(interval, smooth, 'valid')
					ax.plot(np.arange(smooth//2, smooth//2+len(smoothed)),smoothed, linewidth=linewidth)
		
		plt.pause(0.01)  # pause a bit so that plots are updated
		plt.show(block=block)
		

#####################################################################################
# AlgoBase
import inspect

class AlgoBase:
	def __init__(self, create_env_fn, settings):
		self.settings = settings	# should be a Dict
		
		# TODO - make these settings below part of self.settings
		
		# BATCH_SIZE is the number of transitions sampled from the replay buffer
		# GAMMA is the discount factor as mentioned in the previous section
		# EPS_START is the starting value of epsilon
		# EPS_END is the final value of epsilon
		# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
		# TAU is the update rate of the target network
		# LR is the learning rate of the ``AdamW`` optimizer
		self.BATCH_SIZE = 128
		self.GAMMA = 0.99
		self.EPS_START = 0.9
		self.EPS_END = 0.05
		#self.EPS_DECAY = 5000 #1000   # based on steps
		self.EPS_DECAY = 50   # based on episodes
		self.TAU = 0.005
		self.LR = 1e-4
		self.MEM_SIZE = 10000
		
		self.create_env_fn = create_env_fn
		temp_env = create_env_fn()
		self.env_name = temp_env.spec.name
		temp_env.close()
		self.logger = Logger()
		self.memory = AlgoMemory(self.MEM_SIZE)
		self.saver = Saver( self.get_save_name() )
		self.steps_done = 0
		self.episodes_done = 0
		self.average_duration = 0
		
	def get_save_name(self):
		save_name = "ptrl_"+__name__+"_"+self.env_name
		for hidden_layer_size in self.settings["hidden_layer_sizes"]:
			save_name += "_"+str(hidden_layer_size)
		return save_name

	def save(self):
		print(f"{inspect.currentframe().f_code.co_name} overload in child classes")

	def load(self):
		print(f"{inspect.currentframe().f_code.co_name} overload in child classes")
		
	def load_if_save_exists(self):
		if self.saver.save_exists():
			self.load()
		

#####################################################################################
# Settings iterations


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
	
	

