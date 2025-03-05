import ptau_utils as utils
import ptau_model
import ptau_dataset
import ptau_export_spectrograms as spectrograms
from torch.utils.data import DataLoader
import os
import glob
import numpy as np

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
	spectrogram_dbs 			= file_unpack['data']
	spectrograms.show_spectrogram(spectrogram_dbs, block=block)

#view_model_graph("dialog_detect", [ 64, 128, 256, "A" ])
#view_all_model_graphs(block=False)
view_test_results_histogram("dialog_detect", [ 64, 64, 256, "A" ], block=True)
#view_spectrogram('D:/wkspaces/ai_data/ptau/non_dialog/spec_0006786.npz', block=True) #False Positive 98.0%: path=, time_slice=776
#spectrograms.play_audio_from_huggingface("agkphysics/AudioSet", "", 9490)	#found from index.txt