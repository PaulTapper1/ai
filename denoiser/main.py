from audio_denoiser.train import train_model
from audio_denoiser.inference import denoise_audio

print("")
print("")
print("")
print("______________________________________________________")

#train_model()

denoise_audio("data/test/noisy/test_7.wav", save_path="test_7_denoised.wav")