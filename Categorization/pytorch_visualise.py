# pip3 install matplotlib
# pip3 install torch
# pip3 install torchvision

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import datetime
import os
import glob
import shutil
import uuid
import numpy as np

def plot_epoch_graph_from_file(filename, color = "black", color_training_data = ""):
  #print("Loading ",filename)
  
  temp_filename = "temp_load_"+str(uuid.uuid4())+".dat"
  shutil.copyfile(filename, temp_filename)
  checkpoint = torch.load(temp_filename,weights_only=False)
  os.remove(temp_filename)
  
  plt.title(f'{filename} (epochs = {len(checkpoint["graph_epoch"])})')  

  if color_training_data != "":
    plt.plot(checkpoint['graph_epoch'],100-np.array(checkpoint['graph_epoch_accuracy_percentage_training_data']), color = color_training_data, label = f"training data ({checkpoint['graph_epoch_accuracy_percentage_training_data'][-1]:0.2f}%)")
  
  plt.plot(checkpoint['graph_epoch'],100-np.array(checkpoint['graph_epoch_accuracy_percentage']), color = color, label = f"test data ({checkpoint['graph_epoch_accuracy_percentage'][-1]:0.2f}%)")
  plt.legend()
  plt.xlabel(f"epoch")
  plt.xscale("log")
  plt.ylabel(f"error percentage")
  plt.yscale("log")

def plot_epoch_graphs_from_files(file_list):
  for filename in file_list:
    color = "gray"
    if filename == file_list[-1]:
      color = "black"
    plot_epoch_graph_from_file(filename, color)

def plot_show():
  plt.xlabel("epoch")
  plt.ylabel(f"accuracy percentage")
  plt.show()

#plot_epoch_graphs_from_files( [ "pytorch_FashionMNIST_64_64_64_.dat",
                                # "pytorch_FashionMNIST_64_64_128_.dat",
                                # "pytorch_FashionMNIST_64_64_256_.dat",
                                # "pytorch_FashionMNIST_64_128_128_.dat",
                              # ] )
#plot_show()

def plot_dynamic():
  while True:
    plt.clf()
    #plot_epoch_graph_from_file("pytorch_MNIST_32_64_128_A_.dat", (1,0,0), (1,0.7,0.7) )
    plot_epoch_graph_from_file("pytorch_MNIST_64_128_256_A_.dat", (1,0,0), (1,0.7,0.7) )
    plt.draw()  # Redraw the current figure
    plt.pause(5)  # Pause for a very short time

plot_dynamic()

def plot_final_accuracy(filename, setting_dimension):
  print("Loading ",filename)
  checkpoint = torch.load(filename,weights_only=False)
  if "settings" in checkpoint:
    settings = checkpoint["settings"]
    graph_epoch_accuracy_percentage = checkpoint['graph_epoch_accuracy_percentage']
    final_accuracy = graph_epoch_accuracy_percentage[-1]
    plot_setting = settings[setting_dimension]
    label = f"{settings}({checkpoint['epoch']})"
    
    plt.plot(plot_setting, final_accuracy, marker='o', markersize=10, color='red')  
    plt.text(plot_setting+2, final_accuracy, label, fontsize=8, color='blue')

def plot_final_accuracy_of_all_graphs():
  matching_files = glob.glob('*.dat')
  setting_dimension = 0

  # Print the list of matching files
  for file_path in matching_files:
      plot_final_accuracy(file_path, setting_dimension)
  
  plt.xlabel(f"setting {setting_dimension}")
  plt.ylabel(f"accuracy percentage")
  plt.show()


plot_final_accuracy_of_all_graphs()