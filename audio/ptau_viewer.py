import ptau_utils as utils
import ptau_model
import ptau_dataset
import ptau_export_spectrograms as spectrograms
import torch
from torch.utils.data import DataLoader
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import time

utils.move_to_data_folder()

def view_model_graph(name_root, settings):
	model = ptau_model.Model(name_root, settings)
	model.plot(block=True)
	
def view_all_model_graphs(block=True):
	ptau_model.Model._plot_start(xmin=50)
	for file in glob.glob("*.logger"):
		name 	= file.replace(".logger","")
		saver 	= utils.Saver(name)
		logger 	= utils.Logger(name)
		saver.load_data_into("logger", 	logger)
		ptau_model.Model._plot_data(logger.data["epoch_error_percentage"], smooth=100, show_unsmoothed=False, label=name)
	ptau_model.Model._plot_end(block=block)

def view_test_results_histogram(name_root, settings, block=True):
	test_data 			= ptau_dataset.SpectrogramDataset("D:/wkspaces/ai_data/ptau", 1)
	test_dataloader 	= DataLoader(test_data, batch_size=utils.batch_size)
	model 				= ptau_model.Model(name_root, settings)
	model.test_loop(test_dataloader, track_confidence=True)
	model.plot_confidence(block=block)

def view_spectrogram(npz_path, block=True):
	file_unpack 		= np.load(npz_path)
	spectrogram_dbs 	= file_unpack['data']
	spectrograms.show_spectrogram(spectrogram_dbs, block=block)

def load_model(name, settings):
	model_name = ptau_model.get_save_name(name, settings)
	saver 	= utils.Saver(model_name)
	model	= ptau_model.NeuralNetwork(settings=settings)
	saver.load_data_into("model", 		model, 	is_net=True	)
	return model

def apply_model_to_spectrograms(name, settings, npz_paths, block=True):
	model 				= load_model(name, settings)
	for npz_path in npz_paths:
		spectrogram_data	= spectrograms.SpectrogramFileData(npz_path)
		dialog_confidence	= [0]*(utils.timeslices_wanted-1)	# buffer up a sub-spectrogram width of zeroes
		time_start = time.time()
		with torch.no_grad():	# slight performance improvement
			for time_offset in np.arange(spectrogram_data.time_slack):
				dialog_confidence.append(model.confidence(spectrogram_data.get_sub_spectrogram(time_offset)))
		time_end = time.time()
		time_per_model = (time_end-time_start)/spectrogram_data.time_slack
		print(f"Took {time_per_model*1000:.2f} ms per model application")
		plt.clf()
		fig = plt.figure(num=1, figsize=(8, 4))
		gs = fig.add_gridspec(2, 1)
		ax = fig.add_subplot(gs[0, 0])
		ax.set_title(f"{ptau_model.get_save_name(name, settings)} applied to {npz_path}", fontsize=8)
		ax.pcolormesh(spectrogram_data.data, cmap="viridis")
		#ax.colorbar(label="Decibels")
		ax = fig.add_subplot(gs[1, 0], sharex=ax)
		ax.set_ylim(0,1)
		ax.plot(dialog_confidence)
		ax.set_ylabel("Dialog confidence")
		plt.pause(0.2)  # pause a bit so that plots are updated
		plt.show(block=block)


def apply_model_to_audio_files(name, settings, audio_file_paths):
	model = load_model(name, settings)

#view_model_graph("dialog_detect", [ 64, 128, 256, "A" ])
#view_all_model_graphs(block=True)
view_test_results_histogram("dialog_detect", [ 64, 64, 64, 256, "A" ], block=True)
#view_spectrogram('D:/wkspaces/ai_data/ptau/non_dialog/spec_0006786.npz', block=True) #False Positive 98.0%: path=, time_slice=776
#spectrograms.play_audio_from_huggingface("agkphysics/AudioSet", "", 9490)	#found from index.txt
#apply_model_to_audio_file("dialog_detect", [ 64, 64, 256, "A" ], "somefile.wav")
# apply_model_to_spectrograms("dialog_detect", [ 64, 64, 256, "A" ], 
	# [
		# "D:/wkspaces/ai_data/ptau/non_dialog/spec_0006786.npz",
		# "D:/wkspaces/ai_data/ptau/non_dialog/spec_0006787.npz",
		# "D:/wkspaces/ai_data/ptau/dialog/spec_0006786.npz",
		# "D:/wkspaces/ai_data/ptau/dialog/spec_0006787.npz",
	# ])
# Note - dialog_detect_64_64_256_A_ has Number of parameters = 6,591,938, and takes approx 3.2 ms per application.  
# This is per 512 samples = 10 ms at 48,000, so offline would be about 3x realtime.  Could just do it every 100 ms to give 30x realtime (so a 90min program takes 3mins to process)

#TODO: check audio content which is problematic for Fraunhofer