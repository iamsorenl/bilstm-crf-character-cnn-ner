import torch
import torch.optim as optim
from model import BiLSTM_CRF
from train import train
from data import get_data_loader, word_vocab, tag_vocab, save_predictions
from config import DEVICE
from helper import hamming_loss, convert_batch_sequence

def pipeline(
    name="model",
    emb_dim=5,
    hidden_dim=4,
    epoch_num=2,
    batch_size=2,
    lr=0.01,
    lamb=1e-4,
    char_emb_dim=4,
    char_cnn_stride=2,
    char_cnn_kernel=2,
    char_cnn=False,
    loss="log_loss",
    resume=False,
    cost_val=10,
):

    # Load Data
    train_loader = get_data_loader(batch_size=batch_size, set_name="train")
    dev_loader = get_data_loader(batch_size=batch_size, set_name="dev")
    test_loader = get_data_loader(batch_size=batch_size, set_name="test")

    print(f"Initializing Model: {name} | Loss: {loss} | CharCNN: {char_cnn}")
    
    model = BiLSTM_CRF(
        len(word_vocab),
        tag_vocab.token2idx,
        emb_dim,
        hidden_dim,
        char_cnn=char_cnn,
        char_cnn_stride=char_cnn_stride,
        char_cnn_kernel=char_cnn_kernel,
        char_embedding_dim=char_emb_dim,
        loss=loss,
        cost=hamming_loss(loss_val=cost_val),
    ).to(DEVICE)

    if resume:
        model.load_state_dict(torch.load(f"{name}.pt", map_location=DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)

    print("Training Model...")
    model, train_epoch_times = train(model, optimizer, train_loader, epoch_num, name=name)

    print("Generating Predictions...")

    # Function to generate predictions
    def generate_predictions(loader, filename):
        preds, golds = [], []
        for X, Y, seq_lens, _ in loader:
            _, batch_preds = model.forward(X, seq_lens)
            preds += batch_preds
            golds += Y.cpu().numpy().tolist()

        preds = convert_batch_sequence(preds, tag_vocab)
        golds = convert_batch_sequence(golds, tag_vocab)
        save_predictions(filename, golds, preds)
        print(f"Saved Predictions to {filename}")

    generate_predictions(dev_loader, "A2-Data/dev.predictions")
    generate_predictions(test_loader, "A2-Data/test.predictions")

    print("Output Hyperparameters & Training Time")
    avg_train_time = sum(train_epoch_times) / len(train_epoch_times)
    print(f"Average Training Time per Epoch: {avg_train_time:.2f} seconds")

    return model
