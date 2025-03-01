import torch
import argparse
from config import (
    EMB_DIM, HIDDEN_DIM, EPOCH_NUM, BATCH_SIZE, LR, LAMB, COST_VAL, 
    RESUME, CHAR_EMB_DIM, CHAR_CNN_STRIDE, CHAR_CNN_KERNEL, LOSS_SOFTMAX_MARGIN, 
    LOSS_SVM
)
from pipeline import pipeline

torch.manual_seed(1)

# Model Runner
def run_model(model_name, char_cnn=False, loss_type="log_loss"):
    print(f"Running {model_name} with loss type: {loss_type}")

    pipeline(
        name=model_name,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        epoch_num=EPOCH_NUM,
        batch_size=BATCH_SIZE,
        lr=LR,
        lamb=LAMB,
        cost_val=COST_VAL,
        resume=RESUME,
        char_emb_dim=CHAR_EMB_DIM,
        char_cnn_stride=CHAR_CNN_STRIDE,
        char_cnn_kernel=CHAR_CNN_KERNEL,
        char_cnn=char_cnn,
        loss=loss_type,
    )

def main():
    parser = argparse.ArgumentParser(description="Run BiLSTM-CRF Model Variants")
    parser.add_argument('--model', type=str, choices=[
        'bi_lstm_crf', 
        'bi_lstm_crf_char_cnn', 
        'bi_lstm_crf_svm_loss', 
        'bi_lstm_crf_softmax_margin_loss'
    ], default='bi_lstm_crf', help="Choose model variant (default: bi_lstm_crf)")
    
    parser.add_argument('--model_name', type=str, default='model', help="Name of the model (default: model)")
    
    args = parser.parse_args()

    # Map model choices to configurations
    model_configs = {
        'bi_lstm_crf': (False, "log_loss"),
        'bi_lstm_crf_char_cnn': (True, "log_loss"),
        'bi_lstm_crf_svm_loss': (False, LOSS_SVM),
        'bi_lstm_crf_softmax_margin_loss': (False, LOSS_SOFTMAX_MARGIN),
    }

    char_cnn, loss_type = model_configs[args.model]
    run_model(args.model_name, char_cnn=char_cnn, loss_type=loss_type)

if __name__ == "__main__":
    main()
