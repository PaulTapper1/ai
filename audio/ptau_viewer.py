import ptau_utils as utils
import ptau_model

def view_graph(name_root, settings, block=True):
	model = ptau_model.Model(name_root, settings);
	model.plot(block=block)
	
settings = [ 64, 128, 256, "A" ]
view_graph("dialog_detect", settings)
