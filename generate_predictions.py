import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_preprocessing import read_conll_file, build_vocab, encode_data, NERDataset, collate_fn
from bilstm_crf import BiLSTM_CRF
import sys
from config import EMBEDDING_DIM, CHAR_EMBEDDING_DIM, CHAR_OUT_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE

# ---------------------- Prediction Function ----------------------
def generate_predictions(data_path, model_path, output_path):
    print(f"📂 Loading and preprocessing data from: {data_path}")

    # Step 1: Load data and vocabularies
    data = read_conll_file(data_path)
    token_vocab, char_vocab, label_vocab = build_vocab(data)
    encoded_data = encode_data(data, token_vocab, char_vocab, label_vocab)

    dataset = NERDataset(encoded_data)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn(token_vocab, char_vocab, label_vocab),
        shuffle=False
    )

    # Step 2: Load trained model
    print("🚀 Loading trained model...")
    model = BiLSTM_CRF(
        token_vocab_size=len(token_vocab),
        char_vocab_size=len(char_vocab),
        label_vocab=label_vocab,
        embedding_dim=EMBEDDING_DIM,
        char_embedding_dim=CHAR_EMBEDDING_DIM,
        char_out_dim=CHAR_OUT_DIM,
        hidden_dim=HIDDEN_DIM
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Reverse label_vocab for index-to-label mapping
    idx_to_label = {v: k for k, v in label_vocab.items()}

    # Step 3: Generate predictions and save to file
    print(f"📝 Generating predictions -> {output_path}")
    with open(output_path, "w") as f_out:
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Predicting", leave=False)
            for tokens_padded, chars_padded, labels_padded, lengths in progress_bar:
                _, predictions = model(tokens_padded, chars_padded, lengths)

                for batch_idx, length in enumerate(lengths):
                    pred_tags = predictions[batch_idx][:length]  # Truncate padding
                    for token_idx, tag_idx in enumerate(pred_tags):
                        pred_label = idx_to_label[tag_idx]
                        token = dataset.data[batch_idx][0][token_idx]  # Original token
                        f_out.write(f"{token}\t{pred_label}\n")
                    f_out.write("\n")  # Sentence boundary

    print("✅ Predictions saved successfully.")

# ---------------------- CLI Entry Point ----------------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("\nUsage: python generate_predictions.py <data_path> <model_path> <output_path> <dataset_type>")
        print("\n<dataset_type>: Specify 'dev' or 'test'")
        sys.exit(1)

    data_path, model_path, output_path, dataset_type = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].lower()

    if dataset_type not in ["dev", "test"]:
        print("❌ Error: <dataset_type> must be either 'dev' or 'test'")
        sys.exit(1)

    print(f"🔍 Generating predictions for the {dataset_type} set...")
    generate_predictions(data_path, model_path, output_path)
