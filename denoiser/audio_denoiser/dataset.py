
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
NUM_HOPS = 405 #(3.2 secs)

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




# import torch
# import librosa
# import numpy as np
# from torch.utils.data import Dataset
# from pathlib import Path

# SAMPLE_RATE = 16000
# N_FFT = 512
# HOP_LENGTH = 128
# NUM_HOPS = 401 #(3.2 secs)

# def __fix_length__(data, length=HOP_LENGTH*NUM_HOPS):
    # data_len = len(data)
    # if data_len >= length:
        # return data[:length]
    # return np.pad(data, (0,length-data_len))

# def __fix_length_spec__(data, length=HOP_LENGTH*NUM_HOPS):
    # data_len = data.shape[1]
    # if data_len >= length:
        # return data[:,:length]
    # return np.pad(data, (0,length-data_len))

# class AudioDenoisingDataset(Dataset):
    # def __init__(self, noisy_files, clean_files):
        # self.noisy_files = noisy_files
        # self.clean_files = clean_files

    # def __len__(self):
        # return len(self.noisy_files)

    # def __getitem__(self, idx):
        # noisy_wav_filename = self.noisy_files[idx]
        # clean_wav_filename = self.clean_files[idx]
        
        # noisy_spec_filename = new_filename = Path(noisy_wav_filename).with_suffix(".spec")
        # clean_spec_filename = new_filename = Path(clean_wav_filename).with_suffix(".spec")
        
        # # check for cached spec files
        # if !os.path.exists(noisy_spec_filename) || !os.path.exists(noisy_spec_filename):
            # noisy_wave, _ = librosa.load(noisy_wav_filename, sr=SAMPLE_RATE)
            # clean_wave, _ = librosa.load(clean_wav_filename, sr=SAMPLE_RATE)

            # noisy_spec = librosa.stft(noisy_wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
            # clean_spec = librosa.stft(clean_wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
            
            # np.save(noisy_spec_filename, noisy_spec)
            # np.save(clean_spec_filename, clean_spec)
        # else:
            # noisy_spec = np.load(noisy_spec_filename)
            # clean_spec = np.load(clean_spec_filename)
            
        # noisy_spec = __fix_length_spec__(noisy_spec)
        # clean_spec = __fix_length_spec__(clean_spec)

        # noisy_mag = np.abs(noisy_spec)
        # clean_mag = np.abs(clean_spec)

        # noisy_mag = torch.tensor(noisy_mag, dtype=torch.float32)
        # clean_mag = torch.tensor(clean_mag, dtype=torch.float32)

        # #print(f"AudioDenoisingDataset ({idx}) {noisy_mag.shape}")

        # return noisy_mag.unsqueeze(0), clean_mag.unsqueeze(0)
