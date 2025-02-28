import torch
from torch.nn.utils.rnn import pad_sequence
from config import START_TAG, STOP_TAG

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx  # No need for .item() here

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec, dim=1):
    """
    Computes log-sum-exp in a numerically stable way.
    Args:
        vec (Tensor): shape (..., tagset_size)
        dim (int): Dimension over which to compute log-sum-exp
    Returns:
        Tensor: shape (..., 1) if keepdim=True, else reduced along dim.
    """
    max_score, _ = torch.max(vec, dim=dim, keepdim=True)  # Get max along specified dim
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score), dim=dim, keepdim=True))

def read_conll_file(file_path, with_labels=True):
    data = []
    words, labels = [], []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:  # Sentence boundary (blank line)
                if words:  # Only add if we have words
                    data.append((words, labels) if with_labels else (words, []))
                    words, labels = [], []  # Reset for next sentence
                continue

            parts = line.split("\t")
            if with_labels and len(parts) == 2:
                word, label = parts
                words.append(word)
                labels.append(label)
            elif not with_labels and len(parts) == 1:
                words.append(parts[0])
            else:
                print(f"Warning: Unexpected line format -> {line}")  # Debugging help

    # Handle last sentence (if file does not end with blank line)
    if words:
        data.append((words, labels) if with_labels else (words, []))

    return data

def build_vocab(data):
    word_to_ix = {"<PAD>": 0, "<UNK>": 1}  # Add special tokens
    tag_to_ix = {START_TAG: 0, STOP_TAG: 1}

    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)  # Assign next available index

        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)  # Assign next available index

    return word_to_ix, tag_to_ix

def collate_fn(batch):
    sentences, tags = zip(*batch)  # Unzip batch into sentences and tags

    # Convert to tensors
    sentences = [torch.tensor(sent, dtype=torch.long) for sent in sentences]
    tags = [torch.tensor(tag, dtype=torch.long) for tag in tags]

    # Pad sequences to max length in batch
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)  # Assume <PAD> = 0
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=1)  # Assume <START> = 1

    # Compute original lengths before padding
    lengths = torch.tensor([len(sent) for sent in sentences], dtype=torch.long)

    return padded_sentences, padded_tags, lengths
