# pip3 install matplotlib
# pip3 install torch
# pip3 install torchvision

import sys
def is_colab():
  return 'google.colab' in sys.modules
if is_colab() == False:
  from PTModel import PTModel
  from pytorch_utilities import FashionMNIST_Util
# import torch
# from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import datetime
import os

# # FashionMNIST (low res images of clothing) #############################################################
# training_data = datasets.FashionMNIST(
    # root="data",
    # train=True,
    # download=True,
    # transform=ToTensor()
# )
# test_data = datasets.FashionMNIST(
    # root="data",
    # train=False,
    # download=True,
    # transform=ToTensor()
# )
# MNIST (handwritten digits) #############################################################
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=256) #, num_workers=1, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=256) #, num_workers=1, pin_memory=True)


#epochs = 5
#number_of_runs = 3
#start_hidden_layer_size = 1

#max_epoch = 300
max_epoch = 10000
settings_iterator =  [
    # [16,32,64],         # conv2D layer0
    # [16,32,64],         # conv2D layer1
    # [64,128,256,512],   # linear layer0

    # [64],         # conv2D layer0
    # [64,128],         # conv2D layer1
    # [128,256,512],   # linear layer0

    [64],         # conv2D layer0
    [128],         # conv2D layer1
    [256],         # linear layer0
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
  ]
settings_iterator_cursor = [0,0,0,0]
results = []

while True:
  settings = [ 
                settings_iterator[0][settings_iterator_cursor[0]],
                settings_iterator[1][settings_iterator_cursor[1]],
                settings_iterator[2][settings_iterator_cursor[2]],
                settings_iterator[3][settings_iterator_cursor[3]],
              ]

  ptmodel = PTModel("MNIST", settings );
  now = datetime.datetime.now()
  print(now.strftime("%Y-%m-%d %H:%M:%S"))
  ptmodel.loop_epochs(max_epoch, train_dataloader, test_dataloader)
  
  results.append( [str(settings[0])+","+str(settings[1])+","+str(settings[2]), ptmodel.accuracy_percentage] )  
  results_sorted = dict(sorted(results, key=lambda item: item[1], reverse=True))
  #print(results_sorted)
  for key, value in results_sorted.items():
    print(f"Settings {key}: {value:.2f}%")
  
  exit_loop = False
  cursor_layer = 0
  while True:
    if cursor_layer == len(settings_iterator):
      exit_loop = True
      break
    settings_iterator_cursor[cursor_layer] += 1
    if settings_iterator_cursor[cursor_layer] < len(settings_iterator[cursor_layer]):
      break
    settings_iterator_cursor[cursor_layer] = 0
    cursor_layer += 1
  if exit_loop:
    break

  # plt.plot(ptmodel.graph_epoch, ptmodel.graph_epoch_accuracy_percentage)
  # plt.xlabel("epoch")
  # plt.ylabel(f"accuracy percentage with hidden layer size {hidden_layer_size}")
  # plt.show()




#plt.plot(graph_hidden_layer_size, graph_accuracy_percentage)
#plt.xlabel("hidden layer size")
#plt.ylabel(f"accuracy percentage after {epochs} epochs")
#plt.show()

print("Done!")
