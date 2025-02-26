import torch

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Force initialization of MPS device
        torch.zeros(1).to(device)
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

# Hyperparameters
EMBEDDING_DIM = 100
CHAR_EMBEDDING_DIM = 30
CHAR_OUT_DIM = 50
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
START_TAG = "<START>"
STOP_TAG =  "<STOP>"
DEVICE = device
