import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import START_TAG, STOP_TAG, DEVICE

# -------------------- Utility Functions --------------------

def argmax(vec, dim=1):
    """
    Returns the indices of the maximum values along a specified dimension.
    """
    _, idx = torch.max(vec, dim=dim)
    return idx

def log_sum_exp(vec):
    """
    Computes log-sum-exp in a numerically stable way.
    Args:
        vec (Tensor): shape (batch_size, tagset_size)
    Returns:
        Tensor: shape (batch_size, 1)
    """
    max_score, _ = torch.max(vec, dim=1, keepdim=True)  # (batch_size, 1)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score), dim=1, keepdim=True))  # (batch_size, 1)

def make_vocab(input_data):
    """
    Builds vocabularies for words and tags.
    Args:
        input_data (list): List of (sentence, tags) pairs.
    Returns:
        word_vocab (dict), tag_vocab (dict)
    """
    word_vocab = {"<PAD>": 0, "<UNK>": 1}
    tag_vocab = {START_TAG: 0, STOP_TAG: 1, "<PAD>": 2}

    for sentence, tags in input_data:
        for word in sentence:
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
        for tag in tags:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    return word_vocab, tag_vocab

# -------------------- Dataset Class --------------------

class SentenceDataset(Dataset):
    """
    Dataset for handling tokenized sentences and tags.
    """
    def __init__(self, data, word_to_ix, tag_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]

        # Convert words and tags to indices
        sentence_tensor = torch.tensor(
            [self.word_to_ix.get(word, self.word_to_ix["<UNK>"]) for word in sentence],
            dtype=torch.long
        )
        tag_tensor = torch.tensor(
            [self.tag_to_ix.get(tag, self.tag_to_ix["<PAD>"]) for tag in tags],
            dtype=torch.long
        )

        return sentence_tensor, tag_tensor

# -------------------- Collate Function --------------------

def get_collate_fn(word_to_ix, tag_to_ix):
    """
    Returns a collate function for DataLoader.
    Pads sentences and tags to the length of the longest sequence in the batch.
    """
    def collate_fn(batch):
        sentences, tags = zip(*batch)
        lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)

        # Pad sentences and tags
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=word_to_ix["<PAD>"])
        padded_tags = pad_sequence(tags, batch_first=True, padding_value=tag_to_ix["<PAD>"])

        return padded_sentences, padded_tags, lengths

    return collate_fn
