import random
import matplotlib.pyplot as plt
import os
import glob
import shutil
import uuid
import numpy as np
import json
import rl_utils as rl
import colorsys

if not os.getcwd().endswith("data"):
  os.chdir("data")
  print(F"Set current folder to {os.getcwd()}")

def plot_graph_from_file(filename, value_to_plot, color = "black"):
  #print("Loading ",filename)
  
  temp_filename = "temp_load_"+str(uuid.uuid4())+".json"
  shutil.copyfile(filename, temp_filename)
  with open(temp_filename, 'r') as f:
    load_json = json.load(f)
    meta_state = rl.MetaState.from_dict(load_json['meta_state'])
    #settings = load_json['settings']
  os.remove(temp_filename)

  # # unsmoothed graph
  # plt.plot(meta_state.get_values("episodes"),meta_state.get_values("episode_durations"), color = color, label = f"{filename}")
  
  # smoothed graph
  smooth_size = 30
  box = np.ones(smooth_size) / smooth_size
  smoothed = np.convolve(meta_state.get_values(value_to_plot), box, mode='valid')
  plt.plot(np.arange(smooth_size//2, smooth_size//2+len(smoothed)), smoothed, color = color, label = f"{filename} smoothed")
  plt.legend()
  plt.xlabel(f"episode")
  plt.ylabel(value_to_plot)
  #plt.xscale("log")
  #plt.yscale("log")

color_list = {}

def plot_graph_from_files(block = False):
  file_list = glob.glob('rl_*.json')
  for filename in file_list:
    if not filename in color_list:
      hue = (len(color_list) * 0.17) % 1.0
      color_list[filename] = colorsys.hsv_to_rgb( hue,0.8,0.8)
    #if filename == file_list[-1]:
    #  color = "black"
    #value_to_plot = "episode_durations"
    value_to_plot = "reward_total"
    plot_graph_from_file(filename, value_to_plot, color_list[filename])

  plt.xlabel("Episode")
  plt.ylabel(value_to_plot)
  plt.show(block=block)

def plot_dynamic():
  while True:
    plt.clf()
    plot_graph_from_files()
    plt.draw()  # Redraw the current figure
    plt.pause(5)  # Pause for a very short time

plot_dynamic()
#plot_graph_from_files()

