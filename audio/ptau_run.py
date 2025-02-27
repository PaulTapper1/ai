import ptau_utils as utils
import ptau_dataset
import ptau_model
import torch
from torch.utils.data import DataLoader

training_data 		= ptau_dataset.SpectrogramDataset("D:/wkspaces/ai_data/ptau", 0) #, transform=ToTensor)
test_data 			= ptau_dataset.SpectrogramDataset("D:/wkspaces/ai_data/ptau", 1) #, transform=ToTensor)
train_dataloader 	= DataLoader(training_data, shuffle=True, batch_size=utils.batch_size) #, num_workers=1, pin_memory=True)
test_dataloader 	= DataLoader(test_data, batch_size=utils.batch_size) #, pin_memory=True)

#max_epoch = 300
max_epoch = 3000
name =  "dialog_detect"

experiment = utils.Experiment(name,
	[
		[64,128],		 # conv2D layer0
		[128,256],		 # conv2D layer1
		[256,512],		 # linear layer0
		["A"],
		#["A","B"],
		#["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
	] )

while experiment.iterate():
	model = ptau_model.Model(name, experiment.experiment );
	model.loop_epochs(max_epoch, train_dataloader, test_dataloader)
	experiment.experiment_completed(model.accuracy_percentage)
	experiment.plot()

experiment.plot(Block=True)
print("Done!")
