import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm  # 🚀 Progress bar

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
# Detect if GPU is available
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ---------------------------
# Utility Functions
# ---------------------------

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix.get(w, to_ix["<UNK>"]) for w in seq], dtype=torch.long, device=DEVICE)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    return (max_score + torch.log(torch.sum(torch.exp(vec - max_score)))).unsqueeze(0)

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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000., device=DEVICE)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas
        for feat in feats:
            alphas_t = [
                log_sum_exp(forward_var + self.transitions[next_tag] + feat[next_tag])
                for next_tag in range(self.tagset_size)
            ]
            forward_var = torch.cat(alphas_t).view(1, -1)  # No more error here
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        return log_sum_exp(terminal_var)

    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence).unsqueeze(1)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        return self.hidden2tag(lstm_out)

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, device=DEVICE)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=DEVICE), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        return score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        return self._forward_alg(feats) - self._score_sentence(feats, tags)

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        return lstm_feats

# ---------------------------
# Data Processing
# ---------------------------

def process_data(file, train=False):
    data = []
    sentence, tags = [], []

    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":  # Sentence boundary
                if sentence:
                    data.append((sentence, tags if train else []))
                    sentence, tags = [], []
            else:
                parts = line.split('\t')
                sentence.append(parts[0])
                if train and len(parts) == 2:
                    tags.append(parts[1])
        
    # Handle last sentence if file doesn't end with a newline
    if sentence:
        data.append((sentence, tags if train else []))

    print(f"✅ Loaded {len(data)} sentences from {file}")
    return data

def build_vocab(data):
    word_to_ix = {"<PAD>": 0, "<UNK>": 1}
    tag_to_ix = {START_TAG: 0, STOP_TAG: 1}
    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return word_to_ix, tag_to_ix

# ---------------------------
# Training Function with Progress Bar
# ---------------------------

def train(model, train_loader, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)
        for sentences, tags, lengths in progress_bar:
            model.zero_grad()
            batch_loss = sum(model.neg_log_likelihood(sentences[i][:lengths[i]], tags[i][:lengths[i]]) for i in range(len(sentences))) / len(sentences)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            progress_bar.set_postfix(loss=batch_loss.item())
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Average Loss: {total_loss / len(train_loader):.4f}")

    return model

# ---------------------------
# Prediction Function
# ---------------------------

def predict(model, data, output_file):
    model.eval()
    with open(output_file, "w") as f:
        with torch.no_grad():
            for sentence, _ in data:
                sentence_tensor = prepare_sequence(sentence, word_to_ix).to(DEVICE)
                lstm_feats = model(sentence_tensor)
                predictions = torch.argmax(lstm_feats, dim=1)

                ix_to_tag = {v: k for k, v in tag_to_ix.items()}
                predicted_tags = [ix_to_tag[idx.item()] for idx in predictions]

                for word, tag in zip(sentence, predicted_tags):
                    f.write(f"{word}\t{tag}\n")
                f.write("\n")
    print(f"✅ Predictions saved to {output_file}")
    
# ---------------------------
# Main Function
# ---------------------------

def main():
    file_paths = {"train": "A2-data/train", "dev": "A2-data/dev", "test": "A2-data/test"}

    # Process data
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

    # Initialize model and optimizer
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Train model
    model = train(model, train_loader, optimizer)

    # 🚀 Generate predictions for dev and test
    predict(model, dev_data, "A2-data/dev.answers")
    predict(model, test_data, "A2-data/test.answers")

if __name__ == "__main__":
    main()
