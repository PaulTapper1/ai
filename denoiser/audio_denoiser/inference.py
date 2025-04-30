
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from audio_denoiser.model import DenoisingAutoencoder

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denoise_audio(file_path, model_path="denoiser_model.pth", visualize=True):
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    noisy_wave, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    noisy_spec = librosa.stft(noisy_wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag = torch.tensor(np.abs(noisy_spec), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        cleaned_mag = model(noisy_mag).squeeze().cpu().numpy()

    phase = np.angle(noisy_spec)
    reconstructed = librosa.istft(cleaned_mag * np.exp(1j * phase), hop_length=HOP_LENGTH)

    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.title("Original (Noisy)")
        plt.plot(noisy_wave)
        plt.subplot(2, 1, 2)
        plt.title("Denoised Output")
        plt.plot(reconstructed)
        plt.tight_layout()
        plt.show()

    return reconstructed
