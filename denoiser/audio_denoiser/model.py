
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        padding = 0
        chans1 = 16
        chans2 = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, chans1, kernel_size=3, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(chans1, chans2, kernel_size=3, stride=2, padding=padding),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(chans2, chans1, kernel_size=4, stride=2, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(chans1, 1, kernel_size=4, stride=2, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        #print(f"DenoisingAutoencoder.forward 1: x = {x.shape}");
        x = self.encoder(x)
        #print(f"DenoisingAutoencoder.forward 2: x = {x.shape}");
        x = self.decoder(x)
        #print(f"DenoisingAutoencoder.forward 3: x = {x.shape}");
        x = x[:,:,:257,:]
        #print(f"DenoisingAutoencoder.forward 4: x = {x.shape}");
        return x
