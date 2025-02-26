import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import START_TAG, STOP_TAG, DEVICE


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=DEVICE)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def make_vocab(input_data):
    word_vocab = {}
    tag_vocab = {START_TAG: 0, STOP_TAG: 1}

    for sentence, tags in input_data:
        for word in sentence:
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
        for tag in tags:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    return word_vocab, tag_vocab

class SentenceDataset(Dataset):
    def __init__(self, data, word_to_ix, tag_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]
        sentence_tensor = torch.tensor(
            [self.word_to_ix.get(word, self.word_to_ix["<UNK>"]) for word in sentence],
            dtype=torch.long
        )
        tag_tensor = torch.tensor(
            [self.tag_to_ix[tag] for tag in tags],
            dtype=torch.long
        )
        return sentence_tensor, tag_tensor

def collate_fn(batch):
    sentences, tags = zip(*batch)
    lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)

    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)
    return padded_sentences, padded_tags, lengths