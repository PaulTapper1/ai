import sounddevice as sd
import torch
import numpy as np
import librosa
import torch.nn.functional as F
from scipy.signal import stft, istft

# Define the model architecture
class DenoisingAutoencoder(torch.nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(257, 257, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(257, 257, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Initialize the model
model = DenoisingAutoencoder()
model.eval()

# Audio configuration
SAMPLING_RATE = 16000
FRAME_SIZE = 1024
HOP_SIZE = 512

# STFT and ISTFT functions
def get_stft(frame):
    _, _, Zxx = stft(frame, fs=SAMPLING_RATE, nperseg=FRAME_SIZE, noverlap=HOP_SIZE)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    return magnitude, phase

def get_istft(magnitude, phase):
    _, x_reconstructed = istft(magnitude * np.exp(1j * phase), fs=SAMPLING_RATE, nperseg=FRAME_SIZE, noverlap=HOP_SIZE)
    return x_reconstructed

# Real-time denoising function
def real_time_denoising(indata, frames, time, status):
    mag, phase = get_stft(indata)
    mag_tensor = torch.tensor(mag, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        denoised_mag = model(mag_tensor)

    denoised_audio = get_istft(denoised_mag.squeeze(0).numpy(), phase)
    sd.play(denoised_audio, SAMPLING_RATE)

# Set up audio stream
stream = sd.InputStream(callback=real_time_denoising, channels=1, samplerate=SAMPLING_RATE, blocksize=FRAME_SIZE)
stream.start()

# Keep the script running
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Real-time denoising stopped.")
