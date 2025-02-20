import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 🆕 Import tqdm for progress bars
from data_preprocessing import read_conll_file, build_vocab, encode_data, NERDataset, collate_fn
from bilstm_crf import BiLSTM_CRF

# ---------------------- Hyperparameters ----------------------
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.01

# ---------------------- Training Function with Progress Bar ----------------------
def train(model, data_loader, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    # Add tqdm for progress bar 🟢
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)

    for tokens_padded, labels_padded, lengths in progress_bar:
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(tokens_padded, labels_padded, lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        
        # Update progress bar with real-time loss 🔥
        progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})

    print(f"✅ Epoch {epoch + 1}/{total_epochs} - Average Loss: {avg_loss:.4f}")

# ---------------------- Evaluation Function with Progress Bar ----------------------
def evaluate(model, data_loader, label_vocab):
    model.eval()
    print("\n🔎 Evaluating Model...")
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for tokens_padded, labels_padded, lengths in progress_bar:
            # Assuming your model's forward returns (score, predictions)
            score, predictions = model(tokens_padded, lengths)
            progress_bar.set_postfix({"Score": f"{score.mean().item():.4f}"})
            
            # Display only the first batch predictions
            print("\n🔢 Predicted Tags:", predictions[0])
            print("🎯 Actual Tags:", labels_padded[0][:lengths[0]].tolist())
            break  # Only evaluating first batch for brevity

# ---------------------- Main Execution ----------------------
def main(train_filepath):
    # Step 1: Read and preprocess data
    print("📂 Loading and preprocessing data...")
    data = read_conll_file(train_filepath)
    token_vocab, label_vocab = build_vocab(data)
    encoded_data = encode_data(data, token_vocab, label_vocab)
    print(f"🔡 Token vocab size: {len(token_vocab)} | 🏷️ Label vocab size: {len(label_vocab)}")

    # Step 2: Create dataset and DataLoader
    dataset = NERDataset(encoded_data)
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn(token_vocab, label_vocab), 
        shuffle=True
    )

    # Step 3: Initialize model and optimizer
    print("\n🚀 Initializing model...")
    model = BiLSTM_CRF(
        token_vocab_size=len(token_vocab),
        label_vocab=label_vocab,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 4: Train the model with progress bar
    print("\n🏋️ Starting training...")
    for epoch in range(EPOCHS):
        train(model, data_loader, optimizer, epoch, EPOCHS)

    # Step 5: Evaluate the model with progress bar
    evaluate(model, data_loader, label_vocab)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <train_filepath>")
    else:
        main(sys.argv[1])
