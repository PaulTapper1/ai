import ptau_utils as utils
import os
from torch.utils.data import Dataset, DataLoader
import torch
import random
import ptau_export_spectrograms
import numpy as np

class SpectrogramFileData():
	def __init__(self):
		self.data = None
		self.time_slack = 0
		self.time_offset = 0

class SpectrogramDataset(Dataset):
	"""Spectrogram dataset for dialog detection.
		
	Phase 0 = trial, phase 1 = test, phase 2 = validate (eg- if you are using a genetic algorithm using the test results)"""

	phase_percentage			= [ 80, 10, 10 ]
	phase_base					= [  0, 80, 90 ]

	def __init__(self, root_dir, phase):
		"""
		Args:
			root_dir	(string): Path to root directory with subfolders "dialog" and "non_dialog"
		"""
		self.root_dir 			= root_dir
		self.phase				= phase
		self.min_num_entries	= -1
		self.category			= {}
		self.category[0]		= self.extract_category("non_dialog")
		self.category[1]		= self.extract_category("dialog")
		self.file_data			= [None, None]

	def extract_category(self, sub_folder):
		ret						= {}
		this_dir				= os.path.join(self.root_dir, sub_folder)
		ret["dir"]				= this_dir
		ret["index_file"] 		= os.path.join(this_dir, "index.txt")
		with open(ret["index_file"], 'r') as fp:
			ret["lines"] = fp.readlines()
			num_lines = len(ret["lines"])
			ret["num_entries"] = num_lines
		assert ret["num_entries"] > 0
		if self.min_num_entries==-1 or ret["num_entries"]<self.min_num_entries:
			self.min_num_entries = ret["num_entries"]
		# print(f"extract_category: '{this_dir}' has {ret['num_entries']} entries")
		return ret
	
	def get_num_items(self, category):
		return (self.category[category]["num_entries"] * SpectrogramDataset.phase_percentage[self.phase])//100
	
	def map_index(self, index):
#		if self.phase>0:
#			print(f"phase = {self.phase}, index = {index}")
#			if index==1135:
#				print(f"About to crash")
		this_phase_percentage =  SpectrogramDataset.phase_percentage[self.phase]
		num_hundreds = (index // this_phase_percentage)
		ret = num_hundreds * 100
		index_remainder = index - num_hundreds * this_phase_percentage
		assert index_remainder>=0 and index_remainder<SpectrogramDataset.phase_percentage[self.phase]
		ret += SpectrogramDataset.phase_base[self.phase] + index_remainder
		return ret
		
	def __len__(self):
		#return self.get_num_items(0) + self.get_num_items(1)
		return 2*self.min_num_entries

	def __getitem__(self, idx):
		this_category = idx % 2
		assert this_category in [0,1]
		have_found_valid_x = False
		spectrogram_db = False
		while not have_found_valid_x:
			num_items = self.get_num_items(this_category)
			unmapped_idx = random.randint(0, num_items-1)	# just discard idx passed in and select a random one of the correct category
			idx = self.map_index(unmapped_idx)
			#with open(self.category[this_category]["index_file"], 'r') as fp:
			index_lines = self.category[this_category]["lines"]
			while idx>=len(index_lines):
				idx -= len(index_lines)	# dodgy hack to ensure it doesn't go out of bounds of array
			index_line = index_lines[idx]
			index = index_line.split(", ")
			filename = index[3].strip()
			path = os.path.join(self.category[this_category]["dir"], filename+".npz")
			loaded_data = np.load(path)
			spectrogram_db = loaded_data['data']
			freqbins, timebins = np.shape(spectrogram_db)
			time_slack = timebins-utils.timeslices_wanted
			if time_slack>=0:
				time_offset = random.randint(0,time_slack)
				spectrogram_db = spectrogram_db[:,time_offset:time_offset+utils.timeslices_wanted]
				## half the freq resolution
				#spectrogram_db = spectrogram_db[:,x.reshape((-1, 2)).sum(axis=1)
				have_found_valid_x = True
				spectrogram_db = spectrogram_db.astype(np.float32)
		ret_x = torch.from_numpy(spectrogram_db).unsqueeze(0)	# unsqueeze an extra dimension to the tensor at the start to represent "one-channel" image
		return (ret_x, this_category)
		#return (ret_x, torch.from_numpy(this_category))

#####################################################################################
if __name__ == '__main__':
	import pathlib
	print(f"Testing {pathlib.Path(__file__)}")
	dataset = SpectrogramDataset("D:/wkspaces/ai_data/ptau", 0)
	print(f"Num items = {len(dataset)}")
	
	for i in range(0, 20):
		idx = random.randint(0,len(dataset)-1)
		sample 			= dataset[idx]
		spectrogram_db 	= sample[0]
		category 		= sample[1]
		ptau_export_spectrograms.show_spectrogram(spectrogram_db, title=f"Index {idx}. Category = {category}")	# display the spectrogram graphically
		utils.wait_for_any_keypress()
		
