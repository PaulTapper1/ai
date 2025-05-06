print("")
print("")
print("")
print("______________________________________________________")

# from audio_denoiser.download_data import download_data
# download_data()

from audio_denoiser.train import train_model
train_model()

# from audio_denoiser.inference import denoise_audio
# denoise_audio("test_7_noisy.wav", save_path="test_7_denoised.wav")
# #denoise_audio("test_7_denoised.wav", save_path="test_7_denoised2.wav")