# see https://huggingface.co/blog/audio-datasets
# https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition

#pip3 install datasets
#pip3 install soundfile
#pip3 install librosa
#pip3 install pydub playsound==1.3.0	# note - older version important!
#pip3 install datasets transformers sounddevice soundfile  # Install necessary libraries

# from datasets import load_dataset
# ds = load_dataset("mozilla-foundation/common_voice_11_0", "ab", trust_remote_code=True)
# print(ds)

import os
HF_HOME = os.environ['HF_HOME']
#print(f"HF_HOME = '{HF_HOME}'")
assert HF_HOME == "D:\\wkspaces\\ai_data\\huggingface", "You should set your HF_HOME directory in Windows environment variables before importing datasets"
PTAU_HOME = "D:\\wkspaces\\ai_data\\ptau"
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sounddevice as sd
import librosa
import librosa.display
from scipy import signal
import ptau_utils as utils


#-----------------------------------------------------------------------------
# Global settings
n_fft 			= utils.n_fft				# size of FFTs
hop_length 		= utils.hop_length			# samples between each FFT slice
min_freq_hz		= utils.min_freq_hz		# minimum frequency in spectrogram	# https://www.dpamicrophones.com/mic-university/background-knowledge/facts-about-speech-intelligibility/
max_freq_hz		= utils.max_freq_hz		# maximum frequency in spectrogram
num_freq_bins 	= utils.num_freq_bins		# num frequncy bins (distributed logarithmically in frequency range)
max_per_dataset	= 30000

export_count		= [ 11364, 210822 ]	# non dialog, dialog
sub_folder_name		= [ "non_dialog", "dialog" ]

#-----------------------------------------------------------------------------
# Load data
def load_from_huggingface(resource, language):
	# see https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
	dataset = load_dataset(resource, language, split="train", trust_remote_code=True)
	print(f"{resource} '{language}': rows = {len(dataset)}")
	#print(f"num_rows={len(dataset)}")
	#print(dataset)
	#print(dataset[0])
	#print(dataset[0]["audio"])
	return dataset

def make_freq_logscale():	
	scale = np.linspace(math.log(min_freq_hz), math.log(max_freq_hz), num_freq_bins, False)
	scale = np.exp(scale)
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

def get_spectrogram(dataset_item):
	audio_clip = dataset_item["audio"]
	audio_data = audio_clip["array"]
	sample_rate = audio_clip["sampling_rate"]
	sr = sample_rate

	# Calculate the Short-Time Fourier Transform (STFT)
	stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)	# https://librosa.org/doc/main/generated/librosa.stft.html
	spectrogram = np.abs(stft)	# magnitude spectrogram

	# convert into spectrogram with log scale of frequencies
	freq_logscale = make_freq_logscale()
	spectrogram = convert_spectrogram_to_freq_scale(spectrogram, sample_rate, freq_logscale)
	
	# convert spectrogram to have dB values
	spectrogram = np.fmax(spectrogram,1e-8)
	spectrogram_dbs = 20 * np.log10(spectrogram)	# convert to log (dBs)
	max_dbs = np.max(spectrogram_dbs)
	dbs_floor = -60
	spectrogram_dbs -= max_dbs + dbs_floor
	spectrogram_dbs = np.fmax(spectrogram_dbs, 0)  # limit db range
	spectrogram_dbs *= (-1/dbs_floor)
	# results in the spectrogram being scaled and constrained between 0 and -60 dBs, then scaled to be between 1 and 0

	# trim lead-in and lead-out
	freqbins, timebins = np.shape(spectrogram_dbs)
	min_level_for_timeslice = 0.8
	time_start = 0
	while np.max(spectrogram_dbs[:,time_start]) < min_level_for_timeslice:
		time_start += 1
	time_end = timebins-1
	while np.max(spectrogram_dbs[:,time_end]) < min_level_for_timeslice:
		time_end -= 1
	spectrogram_dbs = spectrogram_dbs[:,time_start:time_end]
	
	return spectrogram_dbs

def show_spectrogram(spectrogram_dbs, sample_rate=48000, block=False, title=""):
	freqbins, timebins = np.shape(spectrogram_dbs)
	plt.clf()
	plt.figure(num=1, figsize=(8, 4))
	plt.pcolormesh(spectrogram_dbs, cmap="viridis")
	#clip_length = timebins * hop_length / sample_rate
	plt.title(title)
	plt.colorbar(label="Decibels")
	plt.pause(0.2)  # pause a bit so that plots are updated
	plt.show(block=block)

def play_audio_from_dataset_item(dataset_item, print_info=False):
	audio_clip = dataset_item["audio"]
	audio_data = audio_clip["array"]
	sample_rate = audio_clip["sampling_rate"]
	if print_info:
		print(f"play_audio_from_dataset_item path='{audio_clip['path']}' {dataset_item}")
	sd.play(audio_data, sample_rate)
	sd.wait()  # Wait for playback to finish

def play_audio_from_huggingface(resource, language, idx):
	dataset = load_from_huggingface(resource, language)
	play_audio_from_dataset_item(dataset[idx], print_info=True)

def visualize_data_item(dataset_item, play_audio=True):
	spectrogram_dbs = get_spectrogram(dataset_item)
	show_spectrogram(spectrogram_dbs)
	if play_audio:
		if "human_labels" in dataset_item:
			print(f"human_labels={dataset_item['human_labels']}")
		play_audio_from_dataset_item(dataset_item)

def visualize_some_data_items(dataset, num_items=10):
	id_base = random.randint(0, len(dataset)-1-num_items)
	for i in range(0, num_items):
		visualize_data_item(dataset[id_base+i])
	plt.show(block=True)

def save_spectrogram(spectrogram_dbs, is_dialog, label):
	is_dialog_index = 1 if is_dialog else 0
	folder = PTAU_HOME+"\\"+sub_folder_name[is_dialog_index]+"\\"
	os.makedirs(folder, exist_ok=True)
	file_num = str(export_count[is_dialog_index]).zfill(7)
	filename = f"spec_{file_num}"
	with open(folder+'index.txt', 'a+') as file:
		file.write(label+filename+"\n")
	np.savez_compressed(f"{folder}{filename}.npz", data=spectrogram_dbs)
	export_count[is_dialog_index] += 1
	# # To load the array later:
	# loaded_data = np.load('my_compressed_array.npz')
	# loaded_array = loaded_data['data']

def export_spectrogram_from_dataset_item(dataset_item, is_dialog, label):
	if not is_dialog:
		if "human_labels" in dataset_item:
			labels = dataset_item['human_labels']
		if "caption_auditory" in dataset_item:
			labels = ",".join(dataset_item['caption_auditory'])
		discard_labels = ["speech", "singing", "chant", "talk", "talking"]
		labels = labels.lower()
		for discard_label in discard_labels:
			if discard_label in labels:
				# this item was labelled as not dialog, but, actually probably is, so discard item
				return
	spectrogram_dbs = get_spectrogram(dataset_item)
	show_spectrogram(spectrogram_dbs)
	save_spectrogram(spectrogram_dbs, is_dialog, label)

def export_spectrograms_from_dataset(dataset, is_dialog, label):
	max_item = len(dataset)-1
	if max_item > max_per_dataset:
		max_item = max_per_dataset
	for i in range(0, max_item):
		print(f"Progress: {i} / {max_item} ({i/max_item*100:.3f}%) [Total exported = {export_count[0]} non dialog, {export_count[1]} dialog]\r", end="")
		export_spectrogram_from_dataset_item(dataset[i], is_dialog, label+str(i)+", ")
	print("\n")

def process_dataset(resource, language, is_dialog):
	dataset = load_from_huggingface(resource, language)
	label = f"{resource}, {language}, "
	export_spectrograms_from_dataset(dataset, is_dialog, label)
	#visualize_some_data_items(dataset)

class SpectrogramFileData():
	def __init__(self, path, num_per_file=-1):
		self.path 			= path
		file_unpack 		= np.load(path)
		self.data 			= file_unpack['data']
		self.num_left 		= num_per_file
		freqbins, timebins 	= np.shape(self.data)
		self.time_slack 	= timebins-utils.timeslices_wanted
		#print(f"Caching {path}, time_slack = {self.time_slack}")
		
	def is_ready(self):
		return self.num_left!=0 and self.time_slack>=0
	
	def get_sub_spectrogram_with_info(self, time_offset):
		if self.time_slack>=0:
			spectrogram_db = self.data[:,time_offset:time_offset+utils.timeslices_wanted].astype(np.float32)
			self.num_left -= 1
			return self.path, time_offset+utils.timeslices_wanted, spectrogram_db
		return None, None, None

	def get_sub_spectrogram(self, time_offset):
		path, time_slice, spectrogram_db = self.get_sub_spectrogram_with_info(time_offset)
		return spectrogram_db
	
	def get_random_sub_spectrogram_with_info(self):
		if self.time_slack>=0:
			time_offset = random.randint(0, self.time_slack)
			return self.get_sub_spectrogram_with_info(time_offset)
		return None, None, None

if __name__ == '__main__':
	# process_dataset( "mozilla-foundation/common_voice_11_0", "ja"		, True )
	# process_dataset( "agkphysics/AudioSet", ""							, False )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "en"		, True )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "ar"		, True )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "hi"		, True )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "zh-CN"	, True )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "fr"		, True )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "de"		, True )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "ru"		, True )
	# process_dataset( "mozilla-foundation/common_voice_11_0", "es"		, True )
	print("finished")