SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
NUM_HOPS = 402 #(3.2 secs)
BATCH_SIZE = 16
NUM_BATCHES_PER_EPOCH = 256 # 64 # 
EPOCHS = 10000
NUM_BATCHES_PER_TEST = 823//BATCH_SIZE + 1
SAVE_NAME = "denoiser"
LEARNING_RATE = 1e-3
#LEARNING_RATE = 1e-4
