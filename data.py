import torch
from collections import Counter
import random
from random import sample
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import START_TAG, STOP_TAG, PADDING, UNK_TOKEN, DEVICE

random.seed(1)

# ===========================
# Classes
# ===========================

class NERDataset(Dataset):
    """Custom Dataset class for Named Entity Recognition (NER) tasks."""
    def __init__(self, data):
        self.X = [sent for sent, tag in data if len(sent) > 0]
        self.y = [tag for sent, tag in data if len(sent) > 0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], index

class Vocab:
    """Handles vocabulary for tokens."""
    def __init__(self, tokens, base_map={}, max_size=None):
        self.token2idx = base_map
        self.freq = Counter([token for sequence in tokens for token in sequence])

        vocab_size = 0
        for word, _ in sorted(self.freq.items(), key=lambda item: item[1], reverse=True):
            if max_size is not None and vocab_size > max_size:
                break
            self.insert(word)
            vocab_size += 1

        self.idx2token = reverse_map(self.token2idx)

    def insert(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)

    def lookup_index(self, word):
        return self.token2idx.get(word, self.token2idx[UNK_TOKEN])

    def lookup_token(self, idx):
        return self.idx2token[idx]

    def __len__(self):
        return len(self.token2idx)

    def __repr__(self):
        return str(self.token2idx)

# ===========================
# Functions
# ===========================

def read_data(filename):
    """Reads data into an array of dictionaries (one dictionary per sentence)."""
    data = []
    with open(filename, "r") as f:
        sent = []
        for line in f.readlines():
            if line.strip():
                sent.append(line)
            else:
                data.append(make_data_point(sent))
                sent = []
        if sent:  # Handle last sentence in file
            data.append(make_data_point(sent))
    return data

def make_data_point(sent):
    """Converts a sentence into a dictionary format with tokens and gold tags."""
    sent = [s.strip().split() for s in sent]
    return {"tokens": [s[0] for s in sent], "gold_tags": [s[1] for s in sent]}

def load_data(filename):
    """Loads data from a file and returns a list of token-tag pairs."""
    return [(data["tokens"], data["gold_tags"]) for data in read_data(filename)]

def reverse_map(_map):
    """Reverses a dictionary mapping keys to values."""
    return {val: key for key, val in _map.items()}

def get_max_word_len(word_vocab):
    """Gets the maximum length of words in the vocabulary."""
    return max(len(word) for word in word_vocab.idx2token.values())

def load_datasets():
    """
    Loads and returns the training, development, and test datasets.
    """
    train_data = load_data("A2-data/train")
    dev_data = load_data("A2-data/dev.answers")
    test_data = load_data("A2-data/test_answers/test.answers")

    return NERDataset(train_data), NERDataset(dev_data), NERDataset(test_data)

def load_vocab(train_set):
    """Creates vocabularies for words, tags, and characters from the training dataset."""
    word_vocab = Vocab(train_set.X, base_map={PADDING: 0, UNK_TOKEN: 1})
    tag_vocab = Vocab(
        train_set.y,
        base_map={
            START_TAG: 0,
            STOP_TAG: 1,
            "O": 2,
            "B-DNA": 3,
            "I-DNA": 4,
            "B-RNA": 5,
            "I-RNA": 6,
            "B-protein": 7,
            "I-protein": 8,
            "B-cell_line": 9,
            "I-cell_line": 10,
            "B-cell_type": 11,
            "I-cell_type": 12,
        },
    )
    char_vocab = Vocab(
        [word for sequence in train_set.X for word in sequence],
        base_map={PADDING: 0, UNK_TOKEN: 1},
    )
    max_word_len = get_max_word_len(word_vocab)

    return word_vocab, tag_vocab, char_vocab, max_word_len

# ===========================
# Data Loading Functions
# ===========================

def collate_batch(batch, word_vocab, tag_vocab):
    """Prepares batches for DataLoader with padding and sorting."""
    label_list, text_list, index_list = [], [], []
    
    for _text, _label, _index in batch:
        text_list.append(torch.tensor([word_vocab.lookup_index(token) for token in _text], dtype=torch.long))
        label_list.append(torch.tensor([tag_vocab.lookup_index(tag) for tag in _label], dtype=torch.long))
        index_list.append(torch.tensor(_index, dtype=torch.long))

    len_list = torch.tensor([len(seq) for seq in text_list], dtype=torch.long)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = pad_sequence(label_list, batch_first=True, padding_value=-1)
    index_list = torch.tensor(index_list, dtype=torch.long)

    # Sort the batch according to the sequence length in descending order
    len_list, perm_idx = len_list.sort(0, descending=True)
    text_list, label_list, index_list = text_list[perm_idx], label_list[perm_idx], index_list[perm_idx]

    return text_list.to(DEVICE), label_list.to(DEVICE), len_list, index_list

def get_data_loader(dataset, batch_size=1, word_vocab=None, tag_vocab=None):
    """Returns a DataLoader for the specified dataset."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_batch(batch, word_vocab, tag_vocab),
        shuffle=False,
    )

def get_sampled_data_loader(dataset, batch_size=1, num_samples=20):
    """Returns a DataLoader with a randomly sampled subset of data."""
    sampled_data = sample(dataset, num_samples)
    sampled_dataset = NERDataset(sampled_data)
    return get_data_loader(sampled_dataset, batch_size=batch_size)

def save_predictions(filename, golds, preds):
    """Save predictions to a file in the required format."""
    with open(filename, "w") as f:
        for gold_seq, pred_seq in zip(golds, preds):
            for gold, pred in zip(gold_seq, pred_seq):
                f.write(f"{gold} {pred}\n")
            f.write("\n")  # Sentence separator