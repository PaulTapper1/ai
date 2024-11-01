# code taken originally from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import torch
from torch import nn
#!pip install getch
#import getch
import sys
import select
import matplotlib.pyplot as plt
import random

class FashionMNIST_Util:
    def __init__(self):
      super().__init__()
      
      # # FashionMNIST
      # self.labels_map = {
          # 0: "T-Shirt",
          # 1: "Trouser",
          # 2: "Pullover",
          # 3: "Dress",
          # 4: "Coat",
          # 5: "Sandal",
          # 6: "Shirt",
          # 7: "Sneaker",
          # 8: "Bag",
          # 9: "Ankle Boot",
      # }
      
      # MNIST
      self.labels_map = {
          0: "0",
          1: "1",
          2: "2",
          3: "3",
          4: "4",
          5: "5",
          6: "6",
          7: "7",
          8: "8",
          9: "9",
      }

    def DisplayPrediction(self, _X, _y, _pred):
      #move to cpu in case it was on gpu
      X, y, pred = _X.cpu(), _y.cpu(), _pred.cpu()
       
      incorrect_indices = (pred.argmax(1) != y).nonzero()         # Find the indices of incorrect predictions
      figure = plt.figure(figsize=(3, 3))
      #cols, rows = 3, 3
      #for i in range(1, cols * rows + 1):
      sample_idx = random.choice(incorrect_indices).item()   # Choose a random incorrect prediction index
      img = X[sample_idx]
      label = y[sample_idx].item()

      # Get the corresponding data and prediction
      true_label                  = y[sample_idx].item()
      incorrect_prediction_label  = pred.argmax(1)[sample_idx].item()
      prediction_probabilities    = zip(self.labels_map.values(), pred[sample_idx].softmax(dim=0).tolist())
      prediction_probabilities_sorted = sorted(prediction_probabilities, key=lambda item: item[1], reverse=True)

      # Print the incorrect prediction information
      print("Predicted Label Name:", self.labels_map[incorrect_prediction_label]) # Convert tensor to int
      print("True Label Name:", self.labels_map[true_label]) # Convert tensor to int
  
      for key, value in prediction_probabilities_sorted:
        print(f"{key}: {value*100:.2f}%")
      print("")

      #figure.add_subplot(rows, cols, i)
      #plt.title(self.labels_map[label])
      plt.axis("off")
      plt.imshow(img.squeeze(), cmap="gray")
      plt.show()

      #print("Press any key to see the next incorrect prediction (or 'q' to quit):")
      # key = getch.getch()
      # ## Replace getch.getch() with a platform-specific approach:
      # #if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
      # #    key = sys.stdin.read(1)
      # #else:
      # #    key = '' # or any suitable default value
      # if key == 'q':
        # exit()
      #exit()
