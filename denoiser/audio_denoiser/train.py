
import os
import torch
from torch.utils.data import DataLoader
from audio_denoiser.dataset import AudioDenoisingDataset
from audio_denoiser.model import DenoisingAutoencoder

BATCH_SIZE = 16
NUM_BATCHES_PER_EPOCH = 256 #64
EPOCHS = 1000
SAVE_NAME = "denoiser_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device = {device}")

def train(model, loader, optimizer, criterion):
    model.train()
    num_batches = 0
    for noisy, clean in loader:
        num_batches += 1
        if num_batches == NUM_BATCHES_PER_EPOCH:
            break
        print(f"train: batch {num_batches}/{NUM_BATCHES_PER_EPOCH}\r", end="")
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        #print(f"noisy = {noisy.shape}, clean = {clean.shape}, output = {output.shape}");
        loss = criterion(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def train_model():        
    noisy_files = sorted(os.listdir("data/noisy"))
    clean_files = sorted(os.listdir("data/clean"))
    noisy_paths = [os.path.join("data/noisy", f) for f in noisy_files]
    clean_paths = [os.path.join("data/clean", f) for f in clean_files]

    dataset = AudioDenoisingDataset(noisy_paths, clean_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DenoisingAutoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists(SAVE_NAME):
        print("Found pre-saved model")
        model.load_state_dict(torch.load(SAVE_NAME, map_location=device, weights_only=True))
        model.eval()

    for epoch in range(EPOCHS):
        loss = train(model, loader, optimizer, criterion)
        torch.save(model.state_dict(), SAVE_NAME)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
