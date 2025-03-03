import torch
import argparse
from experiment import experiment

# Set manual seed for reproducibility
torch.manual_seed(1)

def run_experiment(model_name, char_cnn=False, loss_type="log_loss"):
    """Runs the BiLSTM-CRF experiment with specified settings."""
    print(f"Running {model_name} with loss type: {loss_type}")

    experiment(
        emb_dim=100,
        hidden_dim=256,
        epoch_num=15,
        batch_size=16,
        lr=0.005,
        lamb=1e-4,
        name=model_name,
        cost_val=10,
        char_cnn=char_cnn,
        loss=loss_type,
    )

def main():
    """Parse command-line arguments and run the appropriate experiment."""
    parser = argparse.ArgumentParser(description="Run BiLSTM-CRF Model Variants")
    
    # Argument for selecting model variant
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'bi_lstm_crf', 
            'bi_lstm_crf_char_cnn', 
            'bi_lstm_crf_svm_loss', 
            'bi_lstm_crf_softmax_margin_loss'
        ],
        default='bi_lstm_crf',
        help="Choose model variant (default: bi_lstm_crf)"
    )

    # Argument for specifying model name
    parser.add_argument(
        '--model_name',
        type=str,
        default='model',
        help="Name of the model (default: model)"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Define model configurations
    model_configs = {
        'bi_lstm_crf': (False, "log_loss"),
        'bi_lstm_crf_char_cnn': (True, "log_loss"),
        'bi_lstm_crf_svm_loss': (False, "svm_loss"),
        'bi_lstm_crf_softmax_margin_loss': (False, "softmax_margin_loss"),
    }

    # Extract configuration
    char_cnn, loss_type = model_configs[args.model]

    # Run the experiment with selected settings
    run_experiment(args.model_name, char_cnn=char_cnn, loss_type=loss_type)

if __name__ == "__main__":
    main()
