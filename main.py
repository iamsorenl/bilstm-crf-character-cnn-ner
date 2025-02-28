from models.model import BiLSTM_CRF
from utils.helper import prepare_sequence, read_conll_file, build_vocab
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM

torch.manual_seed(1)

def main():
    files = {
        "train": "A2-data/train",
        "dev": "A2-data/dev",
        "test": "A2-data/test"
    }

    # Read data
    train_data = read_conll_file(files["train"])

    # Build vocabulary
    word_to_ix, tag_to_ix = build_vocab(train_data)
    
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(train_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in train_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    for epoch in range(30):  # Small number of epochs for quick test
        for sentence, tags in train_data[:100]:  # Subset to avoid long training
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        print("After training:", model(precheck_sent))



if __name__ == "__main__":
    main()