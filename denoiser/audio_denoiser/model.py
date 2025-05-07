import torch.nn as nn
import numpy as np

#DNN_CHANNELS = [8, 16]
#DNN_CHANNELS = [16, 64]
DNN_CHANNELS = [32, 64]
#DNN_CHANNELS = [16, 32, 64]
#DNN_CHANNELS = [32, 64, 128]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        padding = 0
        chans0 = DNN_CHANNELS[0]
        chans1 = DNN_CHANNELS[1]
        #chans2 = DNN_CHANNELS[2]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, chans0, kernel_size=3, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(chans0, chans1, kernel_size=3, stride=2, padding=padding),
            nn.ReLU(),
            # nn.Conv2d(chans1, chans2, kernel_size=3, stride=2, padding=padding),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(chans2, chans1, kernel_size=4, stride=2, padding=padding),
            # nn.ReLU(),
            nn.ConvTranspose2d(chans1, chans0, kernel_size=4, stride=2, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(chans0, 1, kernel_size=4, stride=2, padding=padding),
            nn.ReLU()
        )
        print(f"Number of parameters = {count_vars(self):,}")


    def forward(self, x0):
        x1 = self.encoder(x0)
        x2 = self.decoder(x1)
        x3 = x2[:,:,:257,:]
        #print(f"DenoisingAutoencoder.forward x0 = {x0.shape}, x1 = {x1.shape}, x2 = {x2.shape}, x3 = {x3.shape}");
        return x3
