import time
import torch
from tqdm import tqdm

from helper import unpad_sequence, convert_batch_sequence
from data import load_vocab, load_datasets, save_predictions

train_set, _, _ = load_datasets()
_, tag_vocab, _, _ = load_vocab(train_set)

EARLY_STOPPING = 3 # Number of epochs to wait for improvement

def train(
    model,
    optimizer,
    train_loader,
    dev_loader,
    epoch_num,
    name="model_name",
    prev_best_score=None,
):
    avg_train_epoch_losses = []
    train_epoch_times = []

    best_loss = float("inf")
    no_improve_count = 0

    train_size = len(train_loader.dataset)

    for epoch in range(1, epoch_num + 1):
        start_time = time.time()
        epoch_train_loss = 0

        # Training Loop
        for X, Y, seq_lens, _ in tqdm(train_loader, desc="Training"):
            model.zero_grad()
            loss = model.neg_log_likelihood(X, Y, seq_lens)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_epoch_loss = epoch_train_loss / train_size
        avg_train_epoch_losses.append(avg_train_epoch_loss)
        end_time = time.time()
        train_epoch_times.append(end_time - start_time)

        # Evaluating and Saving Predictions
        dev_preds = []
        for X, _, seq_lens, _ in tqdm(dev_loader, desc="Validating"):
            _, preds = model.forward(X, seq_lens)
            dev_preds += preds

        # Convert predictions to token sequences
        dev_preds = convert_batch_sequence(dev_preds, tag_vocab)

        # Save predictions
        predictions_filename = f"{name}_dev_predictions.txt"
        save_predictions(predictions_filename, dev_preds)

        # Print loss info
        print(f"Epoch {epoch} Training Loss: {avg_train_epoch_loss}")

        # Early stopping based on loss improvement
        if avg_train_epoch_loss < best_loss:
            best_loss = avg_train_epoch_loss
            no_improve_count = 0
            torch.save(model.state_dict(), f"{name}.pt")
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOPPING:
                print("Not improving, early stopped!!")
                break

    model.load_state_dict(torch.load(f"{name}.pt"))
    return model, train_epoch_times
