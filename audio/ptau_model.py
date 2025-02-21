# code originally taken from https://pytorch.org/tutorials/beginner/basics/intro.html

# pip3 install matplotlib
# pip3 install torch
# pip3 install torchvision

import torch
from torch import nn
import os
import random
import sys
import time
import ptau_utils as utils
import matplotlib.pyplot as plt
import numpy as np

visualize_errors = False
#visualize_errors = True	   # use this to switch off training, and just display visualizations of mis-categorised items

# Note - network is designed to identify whether an input spectrogram is dialog or non dialog
# the spectrogram is set up using the settings from ptau_utils
# input = array of ( num_freq_bins x timeslices_wanted ) floats in range 0..1  ( = 200x30 = 6000 )

# set up neural network
class NeuralNetwork(nn.Module):
	def __init__(self, settings):
		super().__init__()
		#self.flatten = nn.Flatten()
		conv2d_0 = settings[0]
		conv2d_1 = settings[1]
		linear_0 = settings[2]
		self.linear_relu_stack = nn.Sequential(
			nn.Conv2d(1, conv2d_0, kernel_size=3, padding=1),  # Input: 1 channel (grayscale), Output: conv2d_0 channels
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
			nn.Conv2d(conv2d_0, conv2d_1, kernel_size=3, padding=1),  # Input: conv2d_0 channels, Output: conv2d_1 channels
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),   # Downsample by 2 again
			nn.Flatten(),
			nn.Linear(conv2d_1 * utils.num_freq_bins//4 * utils.timeslices_wanted//4, linear_0),
			nn.ReLU(),
			nn.Linear(linear_0, 2),
			#nn.LogSoftmax(),
			)
		#print(self.linear_relu_stack)

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self = self.to(device)
		print("Device = ",device)

	def forward(self, x):
		#x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits


class Model:
	def __init__(self, name, settings):
		super().__init__()
		self.model = NeuralNetwork(settings)
		self.name = name
		self.settings = settings
		self.epoch = 0
		self.learning_rate = 1e-3
		#self.learning_rate = 2e-4
		self.loss_fn = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
		self.accuracy_percentage = 0
		self.graph_epoch = []
		self.graph_epoch_accuracy_percentage = []
		self.graph_epoch_accuracy_percentage_training_data = []
		self.load_if_save_file_present()

	def loop_epochs(self, max_epoch, train_dataloader, test_dataloader):
		while self.epoch < max_epoch:
			self.epoch += 1
			epoch_start_time = time.time()

			if visualize_errors != True:
			  self.train_loop(train_dataloader)
			self.test_loop(test_dataloader)
			self.save()

			epoch_end_time = time.time()
			epoch_elapsed_time = epoch_end_time - epoch_start_time
			device = self.get_device()
			print(f"{self.get_save_name()} epoch {self.epoch}... Accuracy: {self.accuracy_percentage:>0.1f}%. Time per epoch = {epoch_elapsed_time:0.1f}s ({device})")
			self.display_graph(self.graph_epoch_accuracy_percentage)

	def train_loop(self, dataloader):
		#print("train_loop")

		# Set the model to training mode - important for batch normalization and dropout layers
		# Unnecessary in this situation but added for best practices
		self.model.train()
		device = self.get_device()
		correct = 0
		count = 0
		training_percentages = []
		train_batches = 16

		for batch, (X, y) in enumerate(dataloader):
			if train_batches == 0:
				break
			train_batches -= 1

			#X, y = X.to(device), y.to(device)
			X = X.to(device)
			y = y.to(device)

			# Compute prediction and loss
			pred = self.model(X)
			loss = self.loss_fn(pred, y)

			this_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
			correct += this_correct
			count += utils.batch_size

			percentage_correct = this_correct * 100 / utils.batch_size
			training_percentages.append(percentage_correct)
			#print(f"Batch {batch}: percentage_correct = {percentage_correct}")
			#self.display_graph(training_percentages)

			# Backpropagation
			loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()

		accuracy_percentage_training_data = correct / count * 100
		self.graph_epoch_accuracy_percentage_training_data.append(accuracy_percentage_training_data)

	def test_loop(self, dataloader):
		#print("test_loop")
		# Set the model to evaluation mode - important for batch normalization and dropout layers
		# Unnecessary in this situation but added for best practices
		self.model.eval()
		num_batches = len(dataloader)
		test_loss, correct = 0, 0
		count = 0
		test_batches = 16

		# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
		# also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
		with torch.no_grad():
			device = self.get_device()
			for X, y in dataloader:
				if test_batches == 0:
					break
				test_batches -= 1

				X, y = X.to(device), y.to(device)
				pred = self.model(X)
				test_loss += self.loss_fn(pred, y).item()
				this_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
				correct += this_correct
				count += utils.batch_size

		test_loss /= num_batches
		correct /= count
		self.accuracy_percentage = 100*correct
		self.graph_epoch.append(self.epoch)
		self.graph_epoch_accuracy_percentage.append(self.accuracy_percentage)

	def get_device(self):
		return next(self.model.parameters()).device

	def get_save_name(self):
		filename = self.name+"_"
		for setting in self.settings:
		  filename += str(setting)+"_"
		return filename+".dat"

	def save(self):
		filename = self.get_save_name()
		torch.save({
					'settings': self.settings,
					'epoch': self.epoch,
					'model_state_dict': self.model.state_dict(),
					'graph_epoch': self.graph_epoch,
					'graph_epoch_accuracy_percentage': self.graph_epoch_accuracy_percentage,
					'graph_epoch_accuracy_percentage_training_data': self.graph_epoch_accuracy_percentage_training_data,
					}, filename)
		#print("Saved PyTorch Model State to " + filename)

	def load_if_save_file_present(self):
		filename = self.get_save_name()
		if os.path.isfile(filename):
			self.load(filename)
		else:
			print(f"File {filename} not found")

	def load(self, filename):
		print(f"Loading save file {filename}")
		checkpoint = torch.load(filename,weights_only=False)
		self.settings = checkpoint['settings']
		self.epoch = checkpoint['epoch']
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.eval()  # Set the model to evaluation mode
		self.graph_epoch = checkpoint['graph_epoch']
		self.graph_epoch_accuracy_percentage = checkpoint['graph_epoch_accuracy_percentage']
		self.graph_epoch_accuracy_percentage_training_data = checkpoint['graph_epoch_accuracy_percentage_training_data']
		self.accuracy_percentage = self.graph_epoch_accuracy_percentage[-1]
		print(f"Loaded epoch {self.epoch}")

	def display_graph(self, data, smooth = 10, block=False):
		plt.figure(num=0)
		plt.clf()
		plt.plot(data)
		if smooth>0 and len(data) >= smooth:
			window = np.ones(int(smooth))/float(smooth)
			smoothed = np.convolve(data, window, 'valid')
			plt.plot(np.arange(smooth//2, smooth//2+len(smoothed)),smoothed)
		plt.pause(0.2)  # pause a bit so that plots are updated
		plt.show(block=block)
