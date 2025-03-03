import ptau_utils as utils
import ptau_model
import os
import glob

def view_model_graph(name_root, settings, block=True):
	model = ptau_model.Model(name_root, settings)
	model.plot(block=block)
	
#view_model_graph("dialog_detect", [ 64, 128, 256, "A" ])

def view_all_model_graphs():
	utils.move_to_data_folder()
	runs_found = glob.glob("*.logger")
	ptau_model.Model._plot_start(xmin=50)
	for file in runs_found:
		name 	= file.replace(".logger","")
		saver 	= utils.Saver(name)
		logger 	= utils.Logger(name)
		saver.load_data_into("logger", 	logger)
		ptau_model.Model._plot_data(logger.data["epoch_error_percentage"], smooth=100, show_unsmoothed=False, label=name)
	ptau_model.Model._plot_end(block=True)
		

view_all_model_graphs()