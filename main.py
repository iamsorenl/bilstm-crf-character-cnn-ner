import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import BiLSTM_CRF
from utils.helper import make_vocab, prepare_sequence, SentenceDataset, collate_fn
from utils.data import load_data
from config import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, setup_device

torch.manual_seed(1234)
torch.device(DEVICE)

def train(model, train_loader, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        
        for sentences, tags, lengths in progress_bar:
            sentences, tags, lengths = sentences.to(DEVICE), tags.to(DEVICE), lengths.to(DEVICE)
            model.zero_grad()
            loss = model.neg_log_likelihood(sentences, tags, lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / len(train_loader):.4f}")


def predict(model, data, output_file):
    model.eval()
    ix_to_tag = {v: k for k, v in tags_vocab.items()}
    
    with open(output_file, "w") as f:
        with torch.no_grad():
            for sentence, _ in data:
                sentence_tensor = prepare_sequence(sentence, words_vocab).to(DEVICE)
                feats = model(sentence_tensor)
                predictions = torch.argmax(feats, dim=1)

                for word, pred_tag in zip(sentence, predictions):
                    f.write(f"{word}\t{ix_to_tag[pred_tag.item()]}\n")
                f.write("\n")
    print(f"✅ Predictions saved to {output_file}")

def main():
    files = {
        "train": "A2-data/train",
        "dev": "A2-data/dev",
        "test": "A2-data/test"
    }
    # Load and prepare data
    
    N = 500  # Choose a small number for faster training
    train_data = load_data(files["train"])[:N]
    #train_data = load_data(files["train"])
    
    dev_data = load_data(files["dev"], with_labels=False)
    test_data = load_data(files["test"], with_labels=False)

    global words_vocab, tags_vocab
    words_vocab, tags_vocab = make_vocab(train_data)

    model = BiLSTM_CRF(len(words_vocab), tags_vocab, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train(model, train_data, optimizer)

    predict(model, dev_data, "A2-data/dev.predictions")
    predict(model, test_data, "A2-data/test.predictions")

if __name__ == "__main__":
    main()