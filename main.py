import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_preprocessing import read_conll_file, build_vocab, encode_data, NERDataset, collate_fn
from bilstm_crf import BiLSTM_CRF
from config import EMBEDDING_DIM, CHAR_EMBEDDING_DIM, CHAR_OUT_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE

# ---------------------- Training Function ----------------------
def train(model, data_loader, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)

    for tokens_padded, chars_padded, labels_padded, lengths in progress_bar:
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(tokens_padded, chars_padded, labels_padded, lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})

    print(f"✅ Epoch {epoch + 1}/{total_epochs} - Average Loss: {avg_loss:.4f}")

# ---------------------- Evaluation Function ----------------------
def evaluate(model, data_loader, label_vocab):
    model.eval()
    print("\n🔎 Evaluating Model...")
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for tokens_padded, chars_padded, labels_padded, lengths in progress_bar:
            score, predictions = model(tokens_padded, chars_padded, lengths)
            progress_bar.set_postfix({"Score": f"{score.mean().item():.4f}"})
            print("\n🔢 Predicted Tags:", predictions[0])
            print("🎯 Actual Tags:", labels_padded[0][:lengths[0]].tolist())
            break  # Only display first batch for brevity

# ---------------------- Data Preparation ----------------------
def load_data(filepath, batch_size):
    data = read_conll_file(filepath)
    token_vocab, char_vocab, label_vocab = build_vocab(data)  # Ensure char_vocab is built
    encoded_data = encode_data(data, token_vocab, char_vocab, label_vocab)  # Include char encoding
    dataset = NERDataset(encoded_data)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn(token_vocab, char_vocab, label_vocab),
        shuffle=True
    )
    return data_loader, token_vocab, char_vocab, label_vocab

# ---------------------- Main Execution ----------------------
def main(train_filepath, model_output_path):
    print("📂 Loading and preprocessing training data...")
    train_loader, token_vocab, char_vocab, label_vocab = load_data(train_filepath, BATCH_SIZE)
    print(f"🔡 Token vocab size: {len(token_vocab)} | 🔠 Char vocab size: {len(char_vocab)} | 🏷️ Label vocab size: {len(label_vocab)}")

    print("\n🚀 Initializing model...")
    model = BiLSTM_CRF(
        token_vocab_size=len(token_vocab),
        char_vocab_size=len(char_vocab),
        label_vocab=label_vocab,
        embedding_dim=EMBEDDING_DIM,
        char_embedding_dim=CHAR_EMBEDDING_DIM,
        char_out_dim=CHAR_OUT_DIM,
        hidden_dim=HIDDEN_DIM
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n🏋️ Starting training...")
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, epoch, EPOCHS)

    torch.save(model.state_dict(), model_output_path)
    print(f"\n💾 Model saved to: {model_output_path}")

# ---------------------- CLI Entry Point ----------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("\nUsage: python main.py <train_filepath> <model_output_path>")
    else:
        train_filepath = sys.argv[1]
        model_output_path = sys.argv[2]
        main(train_filepath, model_output_path)
