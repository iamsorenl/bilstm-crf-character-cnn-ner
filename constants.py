import torch

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING = "<PAD>"
UNK_TOKEN = "<UNK>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")
DEVICE = device
