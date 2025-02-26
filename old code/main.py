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

# ---------------------- Prediction Generation Function ----------------------
def generate_predictions(model, data_loader, label_vocab, input_filepath, output_path):
    model.eval()
    idx_to_label = {v: k for k, v in label_vocab.items()}  # Map indices to labels

    print(f"\n📝 Generating predictions -> {output_path}")
    
    # Read original tokens from the input file
    with open(input_filepath, 'r', encoding='utf-8') as f_in:
        raw_sentences = [line.strip() for line in f_in.read().strip().split('\n\n')]
        sentences = [sentence.split('\n') for sentence in raw_sentences]

    with open(output_path, "w", encoding='utf-8') as f_out:
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Predicting", leave=False)
            sentence_idx = 0

            for tokens_padded, chars_padded, labels_padded, lengths in progress_bar:
                _, predictions = model(tokens_padded, chars_padded, lengths)

                for batch_idx, length in enumerate(lengths):
                    pred_tags = predictions[batch_idx][:length]
                    original_tokens = sentences[sentence_idx]

                    for token, tag_idx in zip(original_tokens, pred_tags):
                        pred_label = idx_to_label[tag_idx]
                        f_out.write(f"{token}\t{pred_label}\n")

                    f_out.write("\n")  # Sentence boundary
                    sentence_idx += 1

    print("✅ Predictions saved successfully.")

# ---------------------- Data Preparation ----------------------
def load_data(filepath, batch_size, with_labels=True):
    data = read_conll_file(filepath, with_labels=with_labels)  # Pass the correct flag
    token_vocab, char_vocab, label_vocab = build_vocab(data)
    encoded_data = encode_data(data, token_vocab, char_vocab, label_vocab)
    dataset = NERDataset(encoded_data)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn(token_vocab, char_vocab, label_vocab),
        shuffle=False
    )
    return data_loader, token_vocab, char_vocab, label_vocab

# ---------------------- Main Execution ----------------------
def main(train_filepath, prediction_input_filepath, prediction_output_path):
    print("📂 Loading and preprocessing training data...")
    train_loader, token_vocab, char_vocab, label_vocab = load_data(train_filepath, BATCH_SIZE, with_labels=True)

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

    print("\n🔎 Training complete. Generating predictions...")

    # Load dev/test data for prediction
    prediction_loader, _, _, _ = load_data(prediction_input_filepath, BATCH_SIZE, with_labels=False)
    generate_predictions(model, prediction_loader, label_vocab, prediction_input_filepath, prediction_output_path)

# ---------------------- CLI Entry Point ----------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("\nUsage: python main.py <train_filepath> <prediction_input_filepath> <prediction_output_path>")
        print("Example: python main.py A2-data/train A2-data/dev dev.predictions")
        sys.exit(1)

    train_filepath = sys.argv[1]
    prediction_input_filepath = sys.argv[2]
    prediction_output_path = sys.argv[3]

    main(train_filepath, prediction_input_filepath, prediction_output_path)
