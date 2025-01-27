# see https://huggingface.co/blog/audio-datasets
# https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition

#pip install datasets
#pip install soundfile
#pip install playsound==1.2.2	# note - older version important!

# from datasets import load_dataset
# ds = load_dataset("mozilla-foundation/common_voice_11_0", "ab", trust_remote_code=True)
# print(ds)

import os
HF_HOME = os.environ['HF_HOME']
print(f"HF_HOME = '{HF_HOME}'")
assert HF_HOME == "D:\\wkspaces\\ai_data\\huggingface", "You should set your HF_HOME directory in Windows environment variables before importing datasets"
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import math
from playsound import playsound
import random

#-----------------------------------------------------------------------------
# Load data
def load_common_voice_11(language):
	# see https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
	print(f"Loading common_voice_11 '{language}'")
	dataset = load_dataset("mozilla-foundation/common_voice_11_0", language, split="train", trust_remote_code=True)
	#print(dataset)
	#print(dataset[0])
	#print(dataset[0]["audio"])
	
	return dataset

dataset = load_common_voice_11("en"		)
# dataset = load_common_voice_11("ar"		)
# dataset = load_common_voice_11("hi"		)
# dataset = load_common_voice_11("zh-CN"	)
# dataset = load_common_voice_11("fr"		)
# dataset = load_common_voice_11("de"		)
# dataset = load_common_voice_11("ja"		)
# dataset = load_common_voice_11("ru"		)
# dataset = load_common_voice_11("es"		)

def make_freq_logscale(num_bins, min_freq = 200, max_freq = 8000):	# https://www.dpamicrophones.com/mic-university/background-knowledge/facts-about-speech-intelligibility/
	scale = np.linspace(math.log(min_freq), math.log(max_freq), num_bins, False)
	scale = np.exp(scale)
	#print(scale)
	return scale

def convert_spectrogram_to_freq_scale(spectrogram, sample_rate, new_freq_scale):
	freqbins, timebins = np.shape(spectrogram)
	scale_map = np.zeros(len(new_freq_scale)+1, dtype=int)
	scale_multiplier = freqbins / sample_rate * 2.0
	for i in range(0, len(scale_map)-1):
		scale_map[i] = int(new_freq_scale[i]*scale_multiplier)
		scale_map[len(scale_map)-1] = 2*scale_map[len(scale_map)-2] - scale_map[len(scale_map)-3];
		
	# create spectrogram with new freq bins
	new_spectrogram = np.zeros([len(new_freq_scale), timebins])
	for i in range(0, len(new_freq_scale)):
		sum_start 	= scale_map[i]
		sum_end 	= scale_map[i+1]
		if sum_end==sum_start:
			sum_end += 1
		new_spectrogram[i,:] = np.sum(spectrogram[sum_start:sum_end,:], axis=0)

	return new_spectrogram

def show_spectrogram_scipy(audio_data, sample_rate, block=True):
	sr = sample_rate
	import librosa
	import librosa.display
	import matplotlib.pyplot as plt
	import numpy as np
	from scipy import signal

	# setup
	n_fft = 4096		# size of FFTs
	hop_length = 512	# samples between each FFT slice

	# Calculate the Short-Time Fourier Transform (STFT)
	stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)	# https://librosa.org/doc/main/generated/librosa.stft.html
	spectrogram = np.abs(stft)	# magnitude spectrogram

	# convert into spectrogram with log scale of frequencies
	num_freq_bins = 200
	freq_logscale = make_freq_logscale(num_freq_bins)
	spectrogram = convert_spectrogram_to_freq_scale(spectrogram, sample_rate, freq_logscale)
	
	# convert spectrogram to have dB values
	spectrogram = np.fmax(spectrogram,1e-8)
	spectrogram_dbs = 20 * np.log10(spectrogram)	# convert to log (dBs)
	spectrogram_dbs = np.fmax(spectrogram_dbs, -60)  # limit db range

	plt.clf()
	plt.figure(num=1, figsize=(8, 4))
	plt.pcolormesh(spectrogram_dbs, cmap="viridis")
	freqbins, timebins = np.shape(stft)
	clip_length = timebins * hop_length / sample_rate
	plt.title(f"Length = {clip_length:.3f} secs")
	plt.colorbar(label="Decibels")
	#plt.ylabel('Frequency [Hz]')
	#plt.xlabel('Time [sec]')
	plt.pause(0.1)  # pause a bit so that plots are updated
	plt.show(block=block)

def visualize_data_item(dataset, idx):
	audio_data 	= dataset[idx]["audio"]["array"]
	sample_rate = dataset[idx]["audio"]["sampling_rate"]
	path 		= dataset[idx]["audio"]["path"]
	print(F"index={idx}, path='{path}'")
	show_spectrogram_scipy(audio_data, sample_rate, block=False)
	playsound(path)

# Get the audio data from the first example
for i in range(0, 10):
	idx = random_number = random.randint(0, len(dataset)-1)
	visualize_data_item(dataset, idx)

plt.show(block=True)
