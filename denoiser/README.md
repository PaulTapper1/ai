
https://chatgpt.com/c/67f5367d-760c-800f-8f1c-7717e7a409f9

# Audio Denoising Autoencoder

This is a simple prototype of an AI-based audio denoiser using a convolutional autoencoder.

## Requirements
```
pip install torch torchaudio librosa matplotlib numpy
```

## Download data/clean/
```
python download_data.py
```

## Training
Put your paired `.wav` files in:
- `data/noisy/`
- `data/clean/`

Run training:
```
python train.py
```

## Inference
```python
from inference import denoise_audio
output_wave = denoise_audio("path_to_noisy_file.wav")
```






🧠 How the Code Works
1. Dataset (dataset.py)
Loads .wav files from data/noisy/ and data/clean/

Converts them into magnitude spectrograms using librosa

Returns (noisy_spectrogram, clean_spectrogram) pairs for training

2. Model (model.py)
A convolutional autoencoder:

Encoder compresses the noisy spectrogram

Decoder reconstructs a clean spectrogram

Output is a denoised spectrogram

3. Training (train.py)
Loads all .wav files in data/noisy and data/clean

Trains the autoencoder for 10 epochs

Saves the trained model to denoiser_model.pth

Run it with:

bash
Copy
Edit
python audio_denoiser/train.py
4. Inference (inference.py)
Loads a trained model

Takes in a new noisy .wav file

Outputs a cleaned waveform by:

Predicting a denoised spectrogram

Recombining it with the original phase

Reconstructing audio via inverse STFT

Use it in a script like:

python
Copy
Edit
from audio_denoiser.inference import denoise_audio
denoised_wave = denoise_audio("data/noisy/example.wav")
To save the result:

python
Copy
Edit
import soundfile as sf
sf.write("denoised.wav", denoised_wave, samplerate=16000)









🧠 Core Components of the Real-Time Denoiser
1. Audio Input (Stream)
Captures raw audio using a library like sounddevice or pyaudio.

Breaks audio into overlapping chunks (frames) — e.g., 1024 samples each with 50% overlap.

2. Preprocessing
Each frame is converted into a spectrogram (typically using STFT).

The magnitude spectrogram is normalized or scaled to match what the model was trained on.

Shape: [1, freq_bins, time_steps]

3. Model Inference
The pre-trained denoising autoencoder (or UNet, Conv1D, etc.) processes the input.

It removes noise by predicting a cleaner spectrogram.

In most cases, only the magnitude is denoised, and the original phase is reused.

4. Reconstruction
The cleaned magnitude and the original phase are combined.

Inverse STFT (istft) reconstructs the time-domain waveform.

The result is a denoised audio frame.

5. Audio Output
The output frame is streamed to speakers or saved using a playback/output buffer.

All steps are repeated on each new input frame with minimal latency (in milliseconds).

🧪 Simplified Flow
plaintext
Copy
Edit
[Microphone] → [STFT] → [Model] → [ISTFT] → [Speaker]