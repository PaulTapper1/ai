#####################################################################################
# Global settings

n_fft 				= 4096		# size of FFTs
hop_length 			= 512		# samples between each FFT slice
min_freq_hz			= 200		# minimum frequency in spectrogram	# https://www.dpamicrophones.com/mic-university/background-knowledge/facts-about-speech-intelligibility/
max_freq_hz			= 8000		# maximum frequency in spectrogram
num_freq_bins 		= 200		# num frequncy bins (distributed logarithmically in frequency range)
timeslices_wanted	= 32		# num of timeslices (hops) that are fed into the deep learning network
batch_size 			= 512
train_batches		= 64
test_batches 		= 16
#train_batches		= 16
#test_batches 		= 8

def get_algorithm_lead_time_ms(sample_rate = 48000):
	return ((n_fft + (timeslices_wanted-1)*hop_length ) / 48000) ** 1000


#####################################################################################
# Console input / output
import msvcrt
def print_warning(message):
	print('\033[93mWarning: '+message+'\033[0m')	# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
def print_error(message):
	print('\033[91mError: '+message+'\033[0m')	# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
	exit()
def wait_for_any_keypress():
	msvcrt.getch()
	
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
				self.data[key] = [value]*self.num_frames
		if len(self.data[key]) == self.num_frames:
			self.data[key].append(value)
		else:
			self.data[key][self.num_frames] = value
	
	def get_latest_value(self, key):
		if not key in self.data :
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
				self.data[key].append(self.data[key][self.num_frames-2])
				
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

#####################################################################################
# Saveable
class Saveable():
	def to_saveable(self):
		return self
		
	def from_saveable(self, saveable):
		self = saveable

class SaveableDict(dict, Saveable):
	pass

class SaveableList(list, Saveable):
	pass

#####################################################################################
# Saving and Loading
import os
import uuid
import json
import glob
import time
import torch

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
		#print("Saving temporarily disabled")	# TEMP PNT
		#return

		start_time = time.time()
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
		elapsed_time = time.time() - start_time
		#print(f"Saved {self.filename} ({elapsed_time:0.1f} secs)")
	
	def save_exists(self):
		search_filter = os.getcwd()+"\\"+self.filename+".*"
		found_files = glob.glob(search_filter)
		return (len(found_files) > 0)

	def load_data_into(self, extension, obj, is_net=False):
		#loaded_data = torch.load(os.getcwd()+"\\"+self.filename+"."+extension, weights_only=False)
		loaded_data = torch.load(self.filename+"."+extension, weights_only=False)
		if is_net:
			obj.load_state_dict(loaded_data)
			obj.eval()  					# Set the net to evaluation mode
		else:
			obj.from_saveable( loaded_data )

#####################################################################################
# Experiment
plot_figure_num = 2
import datetime
import matplotlib.pyplot as plt
import numpy as np

class Experiment(Saveable):
	def __init__(self, name, experiment_options):
		self.name = name
		self.experiment_options = experiment_options
		self.iterator_cursor = [0]*len(self.experiment_options)
		self.experiment = self._get_experiment_from_cursor()
		self.saver = Saver(name)
		self.completed_experiments = {}
		if self.saver.save_exists():
			self.saver.load_data_into("experiment", self)
			print(f"Loaded experiment completed_experiments = {self.completed_experiments}")
		global plot_figure_num
		self.plot_figure_num = plot_figure_num
		plot_figure_num += 1
		self.plot()

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
		now = datetime.datetime.now()
		print(f"{now.strftime('%Y-%m-%d %H:%M:%S')}: Experiment '{self.name}' {self.experiment}")
		return ret
	
	def get_experiment_str(self):
		return str(self.experiment)

	def experiment_completed(self, results):
		self.completed_experiments [self.get_experiment_str()] = results
		self.saver.add_data_to_save("experiment", self)
		self.saver.save()
		
	def plot(self, block=False, save_image=False):
		if len(self.completed_experiments)==0:
			print(f"{self.name} has no completed experiments")
		else:
			fontsize = 8
			figure = plt.figure(num=self.plot_figure_num)
			plt.clf()
			plt.title("Experiment: "+self.saver.filename, fontsize=fontsize)
			plt.tick_params(axis='y', which='major', labelsize=8)
			figure.subplots_adjust(left=0.20)
			plt.xticks([90,91,92,93,94,95,96,97,98,99,100])
			plt.axis([90,100,-0.5,len(self.completed_experiments)-0.5])
			for num, (key, value) in enumerate(self.completed_experiments.items()):
				print(f"{num}, {key}, {value}")
				plt.barh(key, value)
					
			plt.pause(0.2)  # pause a bit so that plots are updated
			if save_image:
				image_filename = self.saver.filename+".png"
				plt.savefig(image_filename)
				print(f"Saved image {image_filename}")
			plt.show(block=block)
