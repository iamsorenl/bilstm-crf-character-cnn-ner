import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

torch.manual_seed(1234)

# ---------------------------
# Hyperparameters
# ---------------------------
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_IDX = 0
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
BATCH_SIZE = 16
NUM_EPOCHS = 10

# ---------------------------
# Device Configuration
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------
# Utility Functions
# ---------------------------
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix.get(w, to_ix["<UNK>"]) for w in seq], dtype=torch.long, device=DEVICE)

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentence_tensors = [prepare_sequence(s, word_to_ix) for s in sentences]
    tag_tensors = [torch.tensor([tag_to_ix[t] for t in ts], dtype=torch.long, device=DEVICE) for ts in tags]
    lengths = torch.tensor([len(s) for s in sentences], device=DEVICE)

    padded_sentences = pad_sequence(sentence_tensors, batch_first=True, padding_value=PAD_IDX)
    padded_tags = pad_sequence(tag_tensors, batch_first=True, padding_value=PAD_IDX)
    return padded_sentences, padded_tags, lengths

# ---------------------------
# Dataset Class
# ---------------------------
class SentenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------
# BiLSTM-CRF Model
# ---------------------------
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=2,  # ✅ Multi-layer BiLSTM
            dropout=0.1,   # ✅ Dropout for regularization
            bidirectional=True,
            batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size, device=DEVICE))
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

    def _get_lstm_features(self, sentences, lengths):
        embeds = self.word_embeds(sentences)  # (batch_size, seq_len, embedding_dim)
        packed_embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        return self.hidden2tag(lstm_out)  # (batch_size, seq_len, tagset_size)

    def _forward_alg(self, feats):
        batch_size, seq_len, tagset_size = feats.size()
        forward_var = torch.full((batch_size, tagset_size), -10000., device=feats.device)
        forward_var[:, self.tag_to_ix[START_TAG]] = 0.

        for i in range(seq_len):
            emit_scores = feats[:, i].unsqueeze(2)
            trans_scores = self.transitions.unsqueeze(0)
            scores = forward_var.unsqueeze(1) + emit_scores + trans_scores
            forward_var = torch.logsumexp(scores, dim=2)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0)
        return torch.logsumexp(terminal_var, dim=1)  # (batch_size,)

    def _score_sentence(self, feats, tags, lengths):
        batch_size, seq_len, _ = feats.size()
        score = torch.zeros(batch_size, device=feats.device)
        start_tags = torch.full((batch_size, 1), self.tag_to_ix[START_TAG], dtype=torch.long, device=feats.device)
        tags = torch.cat([start_tags, tags], dim=1)

        for i in range(seq_len):
            current_tag, next_tag = tags[:, i], tags[:, i + 1]
            emit_score = feats[torch.arange(batch_size), i, next_tag]
            trans_score = self.transitions[next_tag, current_tag]
            score += emit_score + trans_score

        last_tag = tags[torch.arange(batch_size), lengths]
        score += self.transitions[self.tag_to_ix[STOP_TAG], last_tag]
        return score

    def neg_log_likelihood(self, sentences, tags, lengths):
        feats = self._get_lstm_features(sentences, lengths)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags, lengths)
        return torch.mean(forward_score - gold_score)

# ---------------------------
# Data Processing Functions
# ---------------------------
def process_data(file, train=False):
    data = []
    sentence, tags = [], []

    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    data.append((sentence, tags if train else []))
                    sentence, tags = [], []
            else:
                parts = line.split('\t')
                sentence.append(parts[0])
                if train and len(parts) == 2:
                    tags.append(parts[1])

    if sentence:
        data.append((sentence, tags if train else []))

    print(f"Loaded {len(data)} sentences from {file}")
    return data

def build_vocab(data):
    word_to_ix, tag_to_ix = {"<PAD>": 0, "<UNK>": 1}, {START_TAG: 0, STOP_TAG: 1}
    for sentence, tags in data:
        for word in sentence:
            word_to_ix.setdefault(word, len(word_to_ix))
        for tag in tags:
            tag_to_ix.setdefault(tag, len(tag_to_ix))
    return word_to_ix, tag_to_ix

# ---------------------------
# Training Loop
# ---------------------------
def train(model, train_loader, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for sentences, tags, lengths in progress_bar:
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(sentences, tags, lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / len(train_loader):.4f}")

# ---------------------------
# Prediction Function
# ---------------------------
def predict(model, data, output_file):
    model.eval()
    with open(output_file, "w") as f:
        with torch.no_grad():
            for sentence, _ in data:
                sentence_tensor = prepare_sequence(sentence, word_to_ix).unsqueeze(0)
                lengths = torch.tensor([len(sentence)], device=DEVICE)
                feats = model._get_lstm_features(sentence_tensor, lengths)
                predictions = torch.argmax(feats.squeeze(0), dim=1)
                ix_to_tag = {v: k for k, v in tag_to_ix.items()}
                for word, tag_idx in zip(sentence, predictions):
                    f.write(f"{word}\t{ix_to_tag[tag_idx.item()]}\n")
                f.write("\n")
    print(f"Predictions saved to {output_file}")

# ---------------------------
# Main Execution
# ---------------------------
def main():
    file_paths = {"train": "A2-data/train", "dev": "A2-data/dev", "test": "A2-data/test"}
    train_data = process_data(file_paths["train"], train=True)
    dev_data = process_data(file_paths["dev"], train=False)
    test_data = process_data(file_paths["test"], train=False)

    global word_to_ix, tag_to_ix
    word_to_ix, tag_to_ix = build_vocab(train_data)

    train_loader = DataLoader(
        SentenceDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    train(model, train_loader, optimizer)
    predict(model, dev_data, "A2-data/dev.predictions")
    predict(model, test_data, "A2-data/test.predictions")

if __name__ == "__main__":
    main()