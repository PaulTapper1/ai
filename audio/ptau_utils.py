
#####################################################################################
# Console output
def print_warning(message):
	print('\033[93mWarning: '+message+'\033[0m')	# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
def print_error(message):
	print('\033[91mError: '+message+'\033[0m')	# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
	exit()

#####################################################################################
# Saveable
class Saveable():
	def to_saveable(self):
		return self
		
	def from_saveable(self, saveable):
		self = saveable

class SaveableDict(dict, Saveable):
	pass

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
		print(f"Saved {self.filename} ({elapsed_time:0.1f} secs)")
	
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
		for num, (key, (data, av)) in enumerate(self.completed_experiments.items()):
			data.sort()
			x_coords = np.arange(len(data))*(0.5/len(data)) + float(num)
			all_points = list(zip(x_coords,data))
			lose_points = [p for p in all_points if p[1] < 200]
			win_points = [p for p in all_points if p[1] >= 200]
			plt.plot( [p[0] for p in lose_points], [p[1] for p in lose_points], "ro", markersize=5)
			plt.plot( [p[0] for p in win_points], [p[1] for p in win_points], "go", markersize=5)
	
		plt.pause(0.2)  # pause a bit so that plots are updated
		if save_image:
			image_filename = self.saver.filename+".png"
			plt.savefig(image_filename)
			print(f"Saved image {image_filename}")
		plt.show(block=block)
