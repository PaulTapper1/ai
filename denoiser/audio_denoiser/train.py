
import os
import torch
from torch.utils.data import DataLoader
from audio_denoiser.dataset import AudioDenoisingDataset
from audio_denoiser.model import DenoisingAutoencoder
from audio_denoiser.viewer import Viewer
import time

BATCH_SIZE = 16
NUM_BATCHES_PER_EPOCH = 256 # 64 # 
EPOCHS = 10000
NUM_BATCHES_PER_TEST = 823//BATCH_SIZE + 1
SAVE_NAME = "denoiser"
LEARNING_RATE = 1e-3
#LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device = {device}")

def train(model, loader, optimizer, criterion):
    model.train()
    num_batches = 0
    for noisy, clean in loader:
        num_batches += 1
        if num_batches == NUM_BATCHES_PER_EPOCH:
            break
        print(f"train: batch {num_batches}/{NUM_BATCHES_PER_EPOCH}               \r", end="")
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        #print(f"noisy = {noisy.shape}, clean = {clean.shape}, output = {output.shape}");
        loss = criterion(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def test(model, loader, optimizer, criterion):
    model.train()
    num_batches = 0
    with torch.no_grad():
        for noisy, clean in loader:
            num_batches += 1
            if num_batches == NUM_BATCHES_PER_TEST:
                break
            print(f"test: batch {num_batches}/{NUM_BATCHES_PER_TEST}               \r", end="")
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
    return loss.item()

def train_model():     
    # prepare train and test data
    print("Load train data")
    train_noisy_files = sorted(os.listdir("data/train/noisy"))
    train_clean_files = sorted(os.listdir("data/train/clean"))
    train_noisy_paths = [os.path.join("data/train/noisy", f) for f in train_noisy_files]
    train_clean_paths = [os.path.join("data/train/clean", f) for f in train_clean_files]
    train_dataset = AudioDenoisingDataset(train_noisy_paths, train_clean_paths)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Load test data")
    test_noisy_files = sorted(os.listdir("data/test/noisy"))
    test_clean_files = sorted(os.listdir("data/test/clean"))
    test_noisy_paths = [os.path.join("data/test/noisy", f) for f in test_noisy_files]
    test_clean_paths = [os.path.join("data/test/clean", f) for f in test_clean_files]
    test_dataset = AudioDenoisingDataset(test_noisy_paths, test_clean_paths)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DenoisingAutoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    save_data = {
                    "epoch" : 0,
                    "test_loss" : [],
                }
    viewer = Viewer()

    if os.path.exists(SAVE_NAME+".mdl"):
        model.load_state_dict(torch.load(SAVE_NAME+".mdl", map_location=device, weights_only=True))
        model.eval()
        save_data = torch.load(SAVE_NAME+".dat", weights_only=True)
        print(f"Found pre-saved model with {save_data['epoch']} epochs")
        viewer.view_data(save_data["test_loss"], "test_loss")

    while save_data["epoch"] < EPOCHS:
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, test_loader, optimizer, criterion)
        save_data["epoch"] += 1
        save_data["test_loss"].append(test_loss)
        torch.save(model.state_dict(), SAVE_NAME+".mdl")
        torch.save(save_data, SAVE_NAME+".dat")
        end_time = time.time()
        time_per_k = (end_time-start_time)*1000/(BATCH_SIZE*(NUM_BATCHES_PER_EPOCH+NUM_BATCHES_PER_TEST))
        print(f"Epoch {save_data['epoch']}/{EPOCHS}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, time per k: {time_per_k:.4f}")
        viewer.view_data(save_data["test_loss"], "test_loss")

if __name__ == "__main__":
    main()
