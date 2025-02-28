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
EMBEDDING_DIM = 100  # Standard for word embeddings
HIDDEN_DIM = 256  # More capacity for BiLSTM to learn patterns
CHAR_EMBEDDING_DIM = 30  # Good for capturing subword information
CHAR_OUT_DIM = 50  # Output of Char-CNN
BATCH_SIZE = 32  # Standard batch size
EPOCHS = 3  # More epochs to converge better
LEARNING_RATE = 0.01  # Lower LR for stability
WEIGHT_DECAY = 1e-4  # Regularization
START_TAG = "<START>"
STOP_TAG = "<STOP>"
