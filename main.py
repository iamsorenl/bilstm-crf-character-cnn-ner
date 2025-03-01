import torch
import argparse
from config import EMB_DIM, HIDDEN_DIM, EPOCH_NUM, BATCH_SIZE, LR, LAMB, COST_VAL, RESUME, CHAR_EMB_DIM, CHAR_CNN_STRIDE, CHAR_CNN_KERNEL, LOSS_SOFTMAX_MARGIN, LOSS_SVM, START_TAG, STOP_TAG, PADDING, UNK_TOKEN, DEVICE

torch.manual_seed(1)

def bi_lstm_crf():
    print("Running bi_lstm_crf model")
    

def bi_lstm_crf_char_cnn():
    print("Running bi_lstm_crf_char_cnn model")

def bi_lstm_crf_svm_loss():
    print("Running bi_lstm_crf_svm_loss model")

def bi_lstm_crf_softmax_margin_loss():
    print("Running bi_lstm_crf_softmax_margin_loss model")

def main():
    parser = argparse.ArgumentParser(description="Choose a model variation to run")
    parser.add_argument('--model', type=str, choices=[
        'bi_lstm_crf', 
        'bi_lstm_crf_char_cnn', 
        'bi_lstm_crf_svm_loss', 
        'bi_lstm_crf_softmax_margin_loss'
    ], default='bi_lstm_crf', help="Model variation to run (default: bi_lstm_crf)")
    
    args = parser.parse_args()
    
    if args.model == 'bi_lstm_crf':
        bi_lstm_crf()
    elif args.model == 'bi_lstm_crf_char_cnn':
        bi_lstm_crf_char_cnn()
    elif args.model == 'bi_lstm_crf_svm_loss':
        bi_lstm_crf_svm_loss()
    elif args.model == 'bi_lstm_crf_softmax_margin_loss':
        bi_lstm_crf_softmax_margin_loss()
    else:
        print("Invalid model choice")

if __name__ == "__main__":
    main()