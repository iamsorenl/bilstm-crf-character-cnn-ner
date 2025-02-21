import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import sys

# Special tags for CRF compatibility
PAD_TAG = "<PAD>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK_TOKEN = "<UNK>"

# ------------------------- Read CoNLL File -------------------------
def read_conll_file(filepath):
    data = []
    tokens, labels = [], []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # End of a sentence
            if not line:
                if tokens:  
                    data.append((tokens, labels))
                    tokens, labels = [], []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    tokens.append(token)
                    labels.append(label)
                else:
                    print(f"Warning: Malformed line - {line}")

        # Add the last sentence if file doesn't end with a blank line
        if tokens:
            data.append((tokens, labels))

    return data

# ------------------------- Build Vocabulary -------------------------
def build_vocab(data, min_freq=1):
    token_counter = Counter()
    char_counter = Counter()  # ✅ Added for character vocabulary
    label_counter = Counter()

    for tokens, labels in data:
        token_counter.update(tokens)
        label_counter.update(labels)
        for token in tokens:
            char_counter.update(token)  # ✅ Add characters to char_counter

    # Token vocabulary
    token_vocab = {PAD_TAG: 0, UNK_TOKEN: 1}
    for token, freq in token_counter.items():
        if freq >= min_freq:
            token_vocab[token] = len(token_vocab)

    # Character vocabulary
    char_vocab = {PAD_TAG: 0, UNK_TOKEN: 1}
    for char, freq in char_counter.items():
        if freq >= min_freq:
            char_vocab[char] = len(char_vocab)

    # Label vocabulary
    label_vocab = {PAD_TAG: 0, START_TAG: 1, STOP_TAG: 2}
    for label in label_counter:
        if label not in label_vocab:
            label_vocab[label] = len(label_vocab)

    return token_vocab, char_vocab, label_vocab  # ✅ Now returns 3 vocabularies

# ------------------------- Encode Data -------------------------
def encode_data(data, token_vocab, char_vocab, label_vocab):
    encoded_data = []

    for idx, (tokens, labels) in enumerate(data):
        token_ids = [token_vocab.get(token, token_vocab[UNK_TOKEN]) for token in tokens]
        label_ids = [label_vocab[label] for label in labels]

        # ✅ Encode characters for each token
        char_ids = [[char_vocab.get(char, char_vocab[UNK_TOKEN]) for char in token] for token in tokens]

        if len(token_ids) != len(label_ids):
            print(f"Warning: Length mismatch at sentence {idx + 1}")
            continue

        encoded_data.append((token_ids, char_ids, label_ids))

    return encoded_data


# ------------------------- Pad Sequences -------------------------
def pad_sequences(batch, token_vocab, char_vocab, label_vocab):
    tokens, chars, labels = zip(*batch)

    # Pad tokens and labels
    token_tensors = [torch.tensor(seq, dtype=torch.long) for seq in tokens]
    label_tensors = [torch.tensor(seq, dtype=torch.long) for seq in labels]
    tokens_padded = pad_sequence(token_tensors, batch_first=True, padding_value=token_vocab[PAD_TAG])
    labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=label_vocab[PAD_TAG])

    # Pad characters
    max_word_len = max(len(char_seq) for sent in chars for char_seq in sent)
    char_tensors = [
        torch.tensor(
            [char_seq + [char_vocab[PAD_TAG]] * (max_word_len - len(char_seq)) for char_seq in sent],
            dtype=torch.long
        )
        for sent in chars
    ]
    chars_padded = pad_sequence(char_tensors, batch_first=True, padding_value=char_vocab[PAD_TAG])

    lengths = torch.tensor([len(seq) for seq in tokens], dtype=torch.long)

    return tokens_padded, chars_padded, labels_padded, lengths

# ------------------------- Collate Function -------------------------
def collate_fn(token_vocab, char_vocab, label_vocab):
    return lambda batch: pad_sequences(batch, token_vocab, char_vocab, label_vocab)

# ------------------------- Dataset Class -------------------------
class NERDataset(Dataset):
    def __init__(self, encoded_data):
        self.data = encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Returns (token_ids, char_ids, label_ids)

# ------------------------- Main Function -------------------------
# main function tests the data on the data file provided
def main(filepath):
    data = read_conll_file(filepath)
    token_vocab, label_vocab = build_vocab(data)
    encoded_data = encode_data(data, token_vocab, label_vocab)

    print(f"Token vocab size: {len(token_vocab)}")
    print(f"Label vocab size: {len(label_vocab)}")

    dataset = NERDataset(encoded_data)
    data_loader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=collate_fn(token_vocab, label_vocab), 
        shuffle=False
    )

    for batch_idx, (tokens_padded, labels_padded, lengths) in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}")
        print("Tokens padded shape:", tokens_padded.shape)
        print("Labels padded shape:", labels_padded.shape)
        print("Sentence lengths:", lengths)
        print("Tokens padded:\n", tokens_padded)
        print("Labels padded:\n", labels_padded)
        break  # Only check the first batch

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing.py <filepath>")
    else:
        main(sys.argv[1])
