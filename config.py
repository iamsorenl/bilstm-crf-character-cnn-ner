import torch

# ===========================
# Common settings for all models
# ===========================

EMB_DIM = 100  # Word embedding dimension size (Recommended: 100)
HIDDEN_DIM = 128  # Hidden layer dimension size in BiLSTM (Recommended: 128)
EPOCH_NUM = 10  # Number of training epochs (Recommended: 10)
BATCH_SIZE = 32  # Number of sentences per batch (Recommended: 32)
LR = 0.01  # Learning rate for the optimizer (Recommended: 0.005)
LAMB = 1e-4  # Regularization parameter (lambda) to prevent overfitting (Recommended: 1e-4)
COST_VAL = 10  # Cost value used in some loss functions (Recommended: 10)
RESUME = False  # Whether to resume training from a checkpoint (Recommended: False)

# ===========================
# Char CNN-specific settings
# ===========================

CHAR_EMB_DIM = 30  # Character embedding dimension size (Recommended: 30)
CHAR_CNN_STRIDE = 2  # Stride value for character-level CNN (Recommended: 2)
CHAR_CNN_KERNEL = 3  # Kernel size for character-level CNN (Recommended: 3)

# ===========================
# Loss function types and their recommended settings
# ===========================

LOSS_SOFTMAX_MARGIN = "softmax_margin_loss"  # Softmax-margin loss for structured prediction (Preferred: Epochs 10, Batch 32, LR 0.005)
LOSS_SVM = "svm_loss"  # SVM-based loss function for sequence tagging (Preferred: Epochs 6, Batch 16, LR 0.01)

# ===========================
# constants
# ===========================
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING = "<PAD>"
UNK_TOKEN = "<UNK>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")