import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import sys

# Tags for CRF compatibility
START_TAG = "<START>"
STOP_TAG = "<STOP>"
O_TAG = "O"  # Use "O" for padding and non-entity tags

# ------------------------- Read CoNLL File -------------------------
def read_conll_file(filepath, with_labels=True):
    data = []
    tokens, labels = [], []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()

            if not line:  # Sentence boundary
                if tokens:
                    data.append((tokens, labels if with_labels else []))
                    tokens, labels = [], []
                continue

            parts = line.split('\t')

            if with_labels:
                if len(parts) == 2:
                    token, label = parts
                    tokens.append(token.strip())
                    labels.append(label.strip())
                else:
                    print(f"Warning (Line {line_num}): Expected 2 columns but got {len(parts)}")
            else:
                if len(parts) == 1:
                    tokens.append(parts[0].strip())
                else:
                    print(f"Warning (Line {line_num}): Expected 1 column but got {len(parts)}")

    if tokens:
        data.append((tokens, labels if with_labels else []))

    print(f"✅ Loaded {len(data)} sentences from {filepath}")
    return data

# ------------------------- Build Vocabulary -------------------------
def build_vocab(data, min_freq=1):
    token_counter, char_counter, label_counter = Counter(), Counter(), Counter()

    for tokens, labels in data:
        token_counter.update(tokens)
        for token in tokens:
            char_counter.update(token)
        label_counter.update(labels)

    # Token vocab (no UNK, no PAD)
    token_vocab = {token: idx for idx, (token, _) in enumerate(token_counter.items(), start=0)}

    # Char vocab
    char_vocab = {char: idx for idx, (char, _) in enumerate(char_counter.items(), start=0)}

    # Label vocab: O used as padding, plus START and STOP
    label_vocab = {O_TAG: 0, START_TAG: 1, STOP_TAG: 2}
    for label in label_counter:
        if label not in label_vocab:
            label_vocab[label] = len(label_vocab)

    return token_vocab, char_vocab, label_vocab

# ------------------------- Encode Data -------------------------
def encode_data(data, token_vocab, char_vocab, label_vocab):
    encoded_data = []
    for tokens, labels in data:
        token_ids = [token_vocab.get(token, token_vocab.get(O_TAG, 0)) for token in tokens]
        char_ids = [[char_vocab.get(char, 0) for char in token] for token in tokens]
        label_ids = [label_vocab.get(label, label_vocab[O_TAG]) for label in labels] if labels else [label_vocab[O_TAG]] * len(tokens)

        if labels and len(token_ids) != len(label_ids):
            print("Warning: Mismatched token and label lengths.")
            continue

        encoded_data.append((token_ids, char_ids, label_ids))
    return encoded_data

# ------------------------- Pad Sequences -------------------------
def pad_sequences(batch, token_vocab, char_vocab, label_vocab):
    tokens, chars, labels = zip(*batch)

    token_tensors = [torch.tensor(seq, dtype=torch.long) for seq in tokens]
    label_tensors = [torch.tensor(seq, dtype=torch.long) for seq in labels]

    tokens_padded = pad_sequence(token_tensors, batch_first=True, padding_value=0)  # Padding with index 0 (O)
    labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=label_vocab[O_TAG])

    max_word_len = max(len(char_seq) for sent in chars for char_seq in sent)
    char_tensors = [
        torch.tensor(
            [char_seq + [0] * (max_word_len - len(char_seq)) for char_seq in sent],
            dtype=torch.long
        ) for sent in chars
    ]
    chars_padded = pad_sequence(char_tensors, batch_first=True, padding_value=0)

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
        return self.data[idx]

# ------------------------- Main Function (Testing) -------------------------
def main(filepath, with_labels=True):
    data = read_conll_file(filepath, with_labels=with_labels)
    token_vocab, char_vocab, label_vocab = build_vocab(data)
    encoded_data = encode_data(data, token_vocab, char_vocab, label_vocab)

    print(f"Token vocab size: {len(token_vocab)}")
    print(f"Char vocab size: {len(char_vocab)}")
    print(f"Label vocab size: {len(label_vocab)}")

    dataset = NERDataset(encoded_data)
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn(token_vocab, char_vocab, label_vocab),
        shuffle=False
    )

    for batch_idx, (tokens_padded, chars_padded, labels_padded, lengths) in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}")
        print("Tokens padded shape:", tokens_padded.shape)
        print("Chars padded shape:", chars_padded.shape)
        print("Labels padded shape:", labels_padded.shape)
        print("Sentence lengths:", lengths)
        break

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python data_preprocessing.py <filepath> [with_labels (True/False)]")
    else:
        filepath = sys.argv[1]
        with_labels = sys.argv[2].lower() == "true" if len(sys.argv) == 3 else True
        main(filepath, with_labels)
