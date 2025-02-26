# code originally taken from https://pytorch.org/tutorials/beginner/basics/intro.html

# pip3 install matplotlib
# pip3 install torch
# pip3 install torchvision
# pip3 install pyinstrument

import torch
from torch import nn
import os
import random
import sys
import time
import ptau_utils as utils
import matplotlib.pyplot as plt
import numpy as np
#from pyinstrument import Profiler

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
		self.settings = utils.SaveableList(settings)
		self.epoch = 0
		self.learning_rate = 1e-3
		#self.learning_rate = 2e-4
		self.loss_fn = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
		self.accuracy_percentage = 0
		self.logger = utils.Logger(name)
		self.saver = utils.Saver(self.get_save_name())
		if self.saver.save_exists():
			self.load()
		else:
			print(f"Save file for {self.saver.filename} not found")
						
	def loop_epochs(self, max_epoch, train_dataloader, test_dataloader):
		while self.epoch < max_epoch:
			self.epoch += 1
			epoch_start_time = time.time()

			# with Profiler(interval=0.1) as profiler:
				# self.train_loop(train_dataloader)
				# self.test_loop(test_dataloader)
				# self.save()
			# profiler.print()

			self.train_loop(train_dataloader)
			self.test_loop(test_dataloader)
			self.save()
			self.logger.next_frame()

			epoch_end_time = time.time()
			epoch_elapsed_time = epoch_end_time - epoch_start_time
			device = self.get_device()
			print(f"{self.get_save_name()} epoch {self.epoch} Accuracy: {self.accuracy_percentage:>0.1f}%. Time per epoch = {epoch_elapsed_time:0.1f}s ({device})")
			self.plot()

	def train_loop(self, dataloader):
		#print("train_loop")

		# Set the model to training mode - important for batch normalization and dropout layers
		# Unnecessary in this situation but added for best practices
		self.model.train()
		device = self.get_device()
		correct = 0
		count = 0
		training_percentages = []
		batches = utils.train_batches

		for batch, (X, y) in enumerate(dataloader):
			if batches == 0:
				break
			batches -= 1
			print(f"Training: batches left {batches}/{utils.train_batches}           \r", end="")

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

			# Backpropagation
			loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()

		accuracy_percentage_training_data = correct / count * 100
		self.logger.set_frame_value("epoch_error_percentage_training_data", 100-accuracy_percentage_training_data)

	def test_loop(self, dataloader):
		#print("test_loop")
		# Set the model to evaluation mode - important for batch normalization and dropout layers
		# Unnecessary in this situation but added for best practices
		self.model.eval()
		num_batches = len(dataloader)
		test_loss, correct = 0, 0
		count = 0
		batches = utils.test_batches

		# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
		# also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
		with torch.no_grad():
			device = self.get_device()
			for X, y in dataloader:
				if batches == 0:
					break
				batches -= 1
				print(f"Testing: batches left {batches}/{utils.test_batches}          \r", end="")

				X, y = X.to(device), y.to(device)
				pred = self.model(X)
				test_loss += self.loss_fn(pred, y).item()
				this_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
				correct += this_correct
				count += utils.batch_size

		test_loss /= num_batches
		correct /= count
		self.accuracy_percentage = 100*correct
		self.logger.set_frame_value("epoch",self.epoch)
		self.logger.set_frame_value("epoch_error_percentage",100-self.accuracy_percentage)

	def get_device(self):
		return next(self.model.parameters()).device

	def get_save_name(self):
		filename = self.name+"_"
		for setting in self.settings:
		  filename += str(setting)+"_"
		return filename

	def load(self):
		self.saver.load_data_into("model", 		self.model, 	is_net=True	)
		self.saver.load_data_into("logger", 	self.logger					)
		self.saver.load_data_into("settings", 	self.settings				)
		
		self.epoch 					= self.logger.get_latest_value("epoch")
		self.accuracy_percentage 	= 100-self.logger.get_latest_value("epoch_error_percentage")
		print(f"Loaded model {self.name} epochs = {self.epoch} accuracy = {self.accuracy_percentage}")

	def save(self):
		self.saver.add_data_to_save( "model",			self.model, 	is_net=True )
		self.saver.add_data_to_save( "logger",			self.logger 				)
		self.saver.add_data_to_save( "settings", 		self.settings 				)
		self.saver.save()

	def _plot_data(self, data, sub_from_100=False, smooth=10):
		if sub_from_100:
			data = 100-np.array(data)
		plt.plot(data)
		if smooth>0 and len(data) >= smooth:
			window = np.ones(int(smooth))/float(smooth)
			smoothed = np.convolve(data, window, 'valid')
			plt.plot(np.arange(smooth//2, smooth//2+len(smoothed)),smoothed)
		
		
	def plot(self, smooth = 10, block=False):
		data = self.logger.data["epoch_error_percentage"]
		plt.figure(num=0)
		plt.clf()
		plt.yscale('log')
		plt.xscale('log')
		plt.yticks([50,40,30,20,10,5,4,3,2,1])
		plt.axis([1,len(data),1,50])
		plt.grid(axis='both', which='both')
		plt.title(f"{self.get_save_name()}\nepochs = {len(data)}, final error = {data[-1]:.2f}%")

		self._plot_data(data, sub_from_100=False, smooth=smooth)
		plt.pause(0.2)  # pause a bit so that plots are updated
		plt.show(block=block)
