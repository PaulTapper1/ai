print("")
print("")
print("")
print("TEST ______________________________________________________")

from audio_denoiser.inference import denoise_audio
denoise_audio("test_7_noisy.wav", save_path="test_7_denoised.wav")
#denoise_audio("test_7_denoised.wav", save_path="test_7_denoised2.wav")
