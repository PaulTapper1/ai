from datasets import load_dataset

# Load the VoiceBank-DEMAND-16k dataset
dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

# Access the training and testing splits
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Example: Print the first example from the training set
print(train_dataset[0])

import os

# Define directories to save the audio files
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/noisy", exist_ok=True)

import soundfile as sf  # pip install soundfile

# Save training data
for i, sample in enumerate(train_dataset):
	print(f"train_dataset: {i}\r", end="")
	clean_path = f"data/clean/train_{i}.wav"
	noisy_path = f"data/noisy/train_{i}.wav"
	clean_audio = sample["clean"]["array"]
	noisy_audio = sample["noisy"]["array"]
	sampling_rate = sample["clean"]["sampling_rate"]  # both clean and noisy have same rate
	sf.write(clean_path, clean_audio, samplerate=sampling_rate)
	sf.write(noisy_path, noisy_audio, samplerate=sampling_rate)
print("train_dataset completed")

# Save testing data
for i, sample in enumerate(test_dataset):
	print(f"test_dataset: {i}\r", end="")
	clean_path = f"data/clean/test_{i}.wav"
	noisy_path = f"data/noisy/test_{i}.wav"
	clean_audio = sample["clean"]["array"]
	noisy_audio = sample["noisy"]["array"]
	sampling_rate = sample["clean"]["sampling_rate"]  # both clean and noisy have same rate
	sf.write(clean_path, clean_audio, samplerate=sampling_rate)
	sf.write(noisy_path, noisy_audio, samplerate=sampling_rate)
print("test_dataset completed")
