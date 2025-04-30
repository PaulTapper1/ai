
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
NUM_HOPS = 401 #(3.2 secs)

def __fix_length__(data, length=HOP_LENGTH*NUM_HOPS):
    data_len = len(data)
    if data_len >= length:
        return data[:length]
    return np.pad(data, (0,length-data_len))

class AudioDenoisingDataset(Dataset):
    def __init__(self, noisy_files, clean_files):
        self.noisy_files = noisy_files
        self.clean_files = clean_files

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_wave, _ = librosa.load(self.noisy_files[idx], sr=SAMPLE_RATE)
        clean_wave, _ = librosa.load(self.clean_files[idx], sr=SAMPLE_RATE)

        noisy_wave = __fix_length__(noisy_wave)
        clean_wave = __fix_length__(clean_wave)

        noisy_spec = librosa.stft(noisy_wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
        clean_spec = librosa.stft(clean_wave, n_fft=N_FFT, hop_length=HOP_LENGTH)

        noisy_mag = np.abs(noisy_spec)
        clean_mag = np.abs(clean_spec)

        noisy_mag = torch.tensor(noisy_mag, dtype=torch.float32)
        clean_mag = torch.tensor(clean_mag, dtype=torch.float32)

        #print(f"AudioDenoisingDataset ({idx}) {noisy_mag.shape}")

        return noisy_mag.unsqueeze(0), clean_mag.unsqueeze(0)
