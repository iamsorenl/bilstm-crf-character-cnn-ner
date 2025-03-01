import torch

def argmax(vec):
    # Returns the index of the maximum value in the tensor along dimension 1
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    # Converts a sequence of words to a sequence of indices based on a given mapping
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    # Computes the log-sum-exp of the input tensor for numerical stability
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(
        torch.sum(torch.exp(vec - max_score_broadcast))
    )

def unpad_sequence(sequences, seq_lens):
    # Removes padding from a batch of sequences based on their original lengths
    results = []
    for i, seq in enumerate(sequences):
        results += [seq[: seq_lens[i]]]
    return results

def convert_batch_sequence(batch_sequence, vocab):
    # Converts a batch of sequences from indices to tokens using a vocabulary
    return [convert_sequence(sequence, vocab) for sequence in batch_sequence]

def convert_sequence(sequence, vocab):
    # Converts a sequence from indices to tokens using a vocabulary
    return [vocab.idx2token[idx] for idx in sequence]

def convert_to_char_tensor(token_vector, word_vocab, char_vocab, max_word_len):
    # Converts a sequence of word indices to a tensor of character indices with padding
    char_tensor = []
    for idx in token_vector:
        word = word_vocab.lookup_token(idx.item())
        padded_char = word_to_padded_char(word, char_vocab, max_word_len)
        char_tensor.append(padded_char)
    return torch.cat(char_tensor, 0)

def word_to_padded_char(word, char_vocab, max_word_len):
    # Converts a word to a padded sequence of character indices
    processed_chars = [char_vocab.lookup_index(c) for c in word]
    processed_chars = padding_char(processed_chars, max_word_len)
    # batch * channel * sequence length
    processed_chars = torch.tensor(
        processed_chars, dtype=torch.long
    ).unsqueeze(0)
    return processed_chars

def padding_char(char, max_len):
    # Pads a sequence of character indices to a specified maximum length
    while len(char) < max_len:
        char = [0] + char
        if len(char) == max_len:
            break
        char = char + [0]
    return char
