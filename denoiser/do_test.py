print("")
print("")
print("")
print("TEST ______________________________________________________")

from audio_denoiser.inference import denoise_audio
denoise_audio("test_7_noisy.wav", model_path="denoiser_best.mdl", save_path="test_7_denoised.wav")
#denoise_audio("test_7_denoised.wav", model_path="denoiser_best.mdl", save_path="test_7_denoised2.wav")
