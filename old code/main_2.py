import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import json

from models.BiLSTM_CRF import BiLSTM_CRF
from utils.helper import make_vocab, SentenceDataset, get_collate_fn
from utils.data import load_data
from config import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, START_TAG, STOP_TAG

torch.manual_seed(1234)

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

        print(f"✅ Epoch {epoch + 1} - Avg Loss: {total_loss / len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), "bilstm_crf_model.pth")


def predict(model, data_loader, output_file):
    model.eval()
    ix_to_tag = {v: k for k, v in tags_vocab.items() if k not in [START_TAG, STOP_TAG, "<PAD>"]}

    with open(output_file, "w") as f:
        with torch.no_grad():
            for sentences, _, lengths in data_loader:
                sentences = sentences.to(DEVICE)
                predictions = model(sentences, lengths)  # Use the model's forward method

                for i, sentence in enumerate(sentences):
                    seq_len = lengths[i].item()
                    for word_idx, pred_tag_idx in zip(sentence[:seq_len], predictions[i]):
                        word = list(words_vocab.keys())[list(words_vocab.values()).index(word_idx.item())]
                        tag = ix_to_tag[pred_tag_idx]
                        f.write(f"{word}\t{tag}\n")
                    f.write("\n")
    print(f"✅ Predictions saved to {output_file}")

def save_vocab_to_json(vocab, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved vocab to {filepath}")

def main():
    # Predefined file paths
    files = {
        "train": "A2-data/train",
        "dev": "A2-data/dev",
        "test": "A2-data/test"
    }

    # Load and preprocess data
    print("📂 Loading and preprocessing data...")
    #train_data = load_data(files["train"])[:500]  # Use a subset for faster debugging
    train_data = load_data(files["train"])
    dev_data = load_data(files["dev"], with_labels=False)
    test_data = load_data(files["test"], with_labels=False)

    # Build vocabularies
    global words_vocab, tags_vocab
    words_vocab, tags_vocab = make_vocab(train_data)
    print(f"🔡 Word vocab size: {len(words_vocab)}, 🏷️ Tag vocab size: {len(tags_vocab)}")

    # Save vocabularies to JSON
    save_vocab_to_json(words_vocab, "A2-data/word_vocab.json")
    save_vocab_to_json(tags_vocab, "A2-data/tag_vocab.json")

    # Create datasets & dataloaders
    collate = get_collate_fn(words_vocab, tags_vocab)
    train_loader = DataLoader(SentenceDataset(train_data, words_vocab, tags_vocab), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(SentenceDataset(dev_data, words_vocab, tags_vocab), batch_size=BATCH_SIZE, collate_fn=collate)
    test_loader = DataLoader(SentenceDataset(test_data, words_vocab, tags_vocab), batch_size=BATCH_SIZE, collate_fn=collate)

    # Initialize model & optimizer
    model = BiLSTM_CRF(len(words_vocab), tags_vocab, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train and predict
    train(model, train_loader, optimizer)
    predict(model, dev_loader, "A2-data/dev.predictions")
    predict(model, test_loader, "A2-data/test.predictions")


if __name__ == "__main__":
    main()
