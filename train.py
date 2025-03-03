import time

import torch
from tqdm import tqdm

from helper import unpad_sequence, convert_batch_sequence
from data import tag_vocab
from evaluate import batch_evaluate

def train(
    model,
    optimizer,
    train_loader,
    dev_loader,
    epoch_num,
    name="model_name",
    prev_best_score=None,
):
    # record loss in every epoch for train and dev set for plotting
    avg_train_epoch_losses = []
    train_epoch_times = []

    # best score: precision, recall, f-1
    best_score = (float("-inf"), float("-inf"), float("-inf"))
    if prev_best_score:
        best_score = prev_best_score

    train_size = len(train_loader.dataset)
    for epoch in range(1, epoch_num + 1):
        # for recording training time
        # start of training
        start_time = time.time()

        epoch_train_loss = 0

        # training
        for X, Y, seq_lens, _ in tqdm(train_loader, desc="Training"):
            model.zero_grad()

            # Run our forward pass and compute the loss
            loss = model.neg_log_likelihood(X, Y, seq_lens)

            epoch_train_loss += loss.item()
            # Compute gradients with loss
            loss.backward()

            # Update the parameters by optimizer.step()
            optimizer.step()

        # record the average loss
        avg_train_epoch_loss = epoch_train_loss / train_size
        avg_train_epoch_losses.append(avg_train_epoch_loss)

        # end of training
        end_time = time.time()
        train_epoch_times.append(end_time - start_time)

        # Evaluating
        dev_preds = []
        dev_golds = []
        for X, Y, seq_lens, _ in tqdm(dev_loader, desc="Validating"):
            # making prediction on dev set and store the prediction
            _, preds = model.forward(X, seq_lens)
            golds = unpad_sequence(Y.cpu().numpy(), seq_lens)
            dev_preds += preds
            dev_golds += golds

        # evaluate the dev score
        dev_preds = convert_batch_sequence(dev_preds, tag_vocab)
        dev_golds = convert_batch_sequence(dev_golds, tag_vocab)
        dev_precision, dev_recall, dev_f1 = batch_evaluate(
            dev_golds, dev_preds
        )

        # print the performance of current epoch
        print(f"Epoch {epoch} Training Loss: {avg_train_epoch_loss}")
        print(f"Epoch {epoch}  Dev F-1: {dev_f1}")

        # store the best model by evaluating the score
        best_f1 = best_score[2]
        if best_f1 < dev_f1:
            best_score = (dev_precision, dev_recall, dev_f1)
            torch.save(model.state_dict(), f"{name}.pt")

    model.load_state_dict(torch.load(f"{name}.pt"))
    return (
        model,
        train_epoch_times,
    )
