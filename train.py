import time
import torch
from tqdm import tqdm

EARLY_STOPPING = 3  # Stop training if no improvement

def train(
    model,
    optimizer,
    train_loader,
    epoch_num,
    name="model_name",
):
    avg_train_epoch_losses = []
    train_epoch_times = []
    no_improve_count = 0
    train_size = len(train_loader.dataset)

    for epoch in range(1, epoch_num + 1):
        start_time = time.time()
        epoch_train_loss = 0

        # Training Loop
        for X, Y, seq_lens, _ in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            model.zero_grad()
            loss = model.neg_log_likelihood(X, Y, seq_lens)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Record average loss
        avg_train_epoch_loss = epoch_train_loss / train_size
        avg_train_epoch_losses.append(avg_train_epoch_loss)

        end_time = time.time()
        train_epoch_times.append(end_time - start_time)

        print(f"Epoch {epoch} Training Loss: {avg_train_epoch_loss:.4f}")

        # Early stopping logic
        no_improve_count += 1
        if no_improve_count >= EARLY_STOPPING:
            print("Not improving, early stopped!!")
            break

    return model, train_epoch_times
