import torch
import torch.nn as nn
import torch.optim as optim

# Set manual seed for reproducibility
torch.manual_seed(1234)

# Special tags and hyperparameters
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
NUM_EPOCHS = 10
BATCH_SIZE = 16

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, to_ix.setdefault("<UNK>", len(to_ix))) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        return log_sum_exp(terminal_var)

    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        return self.hidden2tag(lstm_out)

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        return score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]

    def _viterbi_decode(self, feats):
        backpointers = []
        forward_var = torch.full((1, self.tagset_size), -10000.)
        forward_var[0][self.tag_to_ix[START_TAG]] = 0

        for feat in feats:
            bptrs_t, viterbivars_t = [], []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.pop()
        best_path.reverse()

        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        return self._forward_alg(feats) - self._score_sentence(feats, tags)

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        return self._viterbi_decode(lstm_feats)

# -------------------------------------------
#               MAIN FUNCTION
# -------------------------------------------
def main():

    # File paths
    file_paths = {
        "train": "A2-data/train",
        "dev": "A2-data/dev",
        "test": "A2-data/test",
        "dev_predictions": "A2-data/dev.predictions",
        "test_predictions": "A2-data/test.predictions",
        "dev_answers": "A2-data/dev.answers",
        "test_answers": "A2-data/test_answers/test.answers"
    }


    # Training Data
    training_data = [
        ("the wall street journal reported today that apple corporation made money".split(),
         "B I I I O O O B I O O".split()),
        ("georgia tech is a university in georgia".split(), "B I O O O O B".split())
    ]

    # Build vocabulary and tag mappings
    word_to_ix = {word: idx for sentence, _ in training_data for idx, word in enumerate(set(sentence))}
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    # Initialize model and optimizer
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Before training predictions
    print("Before Training:")
    with torch.no_grad():
        sentence = prepare_sequence(training_data[0][0], word_to_ix)
        print("Prediction:", model(sentence))

    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss = {total_loss:.4f}")

    # After training predictions
    print("\nAfter Training:")
    with torch.no_grad():
        sentence = prepare_sequence(training_data[0][0], word_to_ix)
        score, tag_seq = model(sentence)
        print("Prediction:", tag_seq)

# -------------------------------------------
#              RUN THE PROGRAM
# -------------------------------------------
if __name__ == "__main__":
    main()
