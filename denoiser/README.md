
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
