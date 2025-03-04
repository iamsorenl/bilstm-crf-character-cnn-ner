import torch.optim as optim

from model import BiLSTM_CRF
from evaluate import (
    inference,
    output_prediction_with_gold_label,
    output_prediction_perl,
    output_hyper_parameters,
    output_training_time,
)
from train import train
from data import (
    get_data_loader,
    word_vocab,
    tag_vocab,
)
from constants import DEVICE
from helper import hamming_loss

word_to_ix = word_vocab.token2idx
tag_to_ix = tag_vocab.token2idx

def experiment(
    emb_dim=100,
    char_emb_dim=50,
    char_cnn_stride=2,
    char_cnn_kernel=3,
    hidden_dim=256,
    epoch_num=10,
    batch_size=16,
    lr=0.005,
    lamb=1e-4,
    name="model",
    char_cnn=False,
    loss="log_loss",
    cost_val=10,
):

    # use data loader for batching data
    train_loader = get_data_loader(batch_size=batch_size, set_name="train")
    dev_loader = get_data_loader(batch_size=batch_size, set_name="dev")
    test_loader = get_data_loader(batch_size=batch_size, set_name="test")

    '''Initialize model'''
    model = BiLSTM_CRF(
        len(word_vocab),
        tag_vocab.token2idx,
        emb_dim,
        hidden_dim,
        char_cnn=char_cnn,
        char_cnn_stride=char_cnn_stride,
        char_cnn_kernel=char_cnn_kernel,
        char_embedding_dim=char_emb_dim,
        loss=loss,
        cost=hamming_loss(loss_val=cost_val),
    ).to(DEVICE)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lamb)

    '''Train model'''
    (model, train_epoch_times) = train(
        model,
        optimizer,
        train_loader,
        dev_loader,
        epoch_num,
        name=name,
    )
    
    '''Evaluate model'''
    dev_all_input, dev_all_preds, dev_all_golds = inference(model, dev_loader)

    test_all_input, test_all_preds, test_all_golds = inference(
        model, test_loader
    )

    '''Output prediction'''
    
    '''Dev Predictions'''
    output_prediction_with_gold_label(
        dev_all_input, 
        dev_all_preds, 
        dev_all_golds, 
        name=f"A2-data/{name}.dev.pred.gold"
    )
    output_prediction_perl(
        dev_all_input,
        dev_all_preds,
        name=f"A2-data/{name}.dev.pred.perl",
    )
    print(f"Output prediction for dev set saved to A2-data/{name}.dev.pred.perl")
    
    '''Test Predictions'''
    output_prediction_with_gold_label(
        test_all_input,
        test_all_preds,
        test_all_golds,
        name=f"A2-data/{name}.test.pred.gold",
    )
    output_prediction_perl(
        test_all_input,
        test_all_preds,
        name=f"A2-data/{name}.test.pred.perl",
    )
    print(f"Output prediction for test set saved to A2-data/{name}.test.pred.perl")

    '''Output hyperparameters'''
    hp_map = {
        "emb_dim": [emb_dim],
        "hidden_dim": [hidden_dim],
        "epoch_num": [epoch_num],
        "batch_size": [batch_size],
        "lr": [lr],
        "lamb": [lamb],
    }
    output_hyper_parameters(hp_map, name=f"{name}.hp")
    output_training_time(
        batch_size,
        sum(train_epoch_times) / len(train_epoch_times),
        name=f"{name}.time",
    )
    print(f"Output hyperparameters saved to {name}.hp")
