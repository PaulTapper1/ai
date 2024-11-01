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

def plot_graph_from_file(filename, color = "black"):
  print("Loading ",filename)
  
  temp_filename = "temp_load_"+str(uuid.uuid4())+".dat"
  shutil.copyfile(filename, temp_filename)
  checkpoint = torch.load(temp_filename,weights_only=False)
  meta_state = checkpoint["meta_state"]
  os.remove(temp_filename)

  # # unsmoothed graph
  # plt.plot(meta_state.get_values("episodes"),meta_state.get_values("episode_durations"), color = color, label = f"{filename}")
  
  # smoothed graph
  smooth_size = 10
  box = np.ones(smooth_size) / smooth_size
  smoothed = np.convolve(meta_state.get_values("episode_durations"), box, mode='same')
  plt.plot(meta_state.get_values("episodes"), smoothed, color = color, label = f"{filename} smoothed")
  plt.legend()
  plt.xlabel(f"episode")
  plt.ylabel(f"duration")
  #plt.xscale("log")
  #plt.yscale("log")

color_list = {}

def plot_graph_from_files():
  file_list = glob.glob('rl_*.dat')
  for filename in file_list:
    if not filename in color_list:
      color_list[filename] = (random.random(),random.random(),random.random())
    #if filename == file_list[-1]:
    #  color = "black"
    plot_graph_from_file(filename, color_list[filename])

  plt.xlabel("Episode")
  plt.ylabel("Duration")
  plt.show()

def plot_dynamic():
  while True:
    plt.clf()
    plot_graph_from_files()
    plt.draw()  # Redraw the current figure
    plt.pause(5)  # Pause for a very short time

#plot_dynamic()
plot_graph_from_files()

