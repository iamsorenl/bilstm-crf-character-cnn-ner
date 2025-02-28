import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.helper import log_sum_exp
from config import START_TAG, STOP_TAG, DEVICE


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # Word embeddings
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Half for bidirectional
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Linear layer to get emission scores
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF transition parameters
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
        #self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000  # No transition to START_TAG
        #self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000  # No transition from STOP_TAG
        # Block illegal transitions:
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000  # No transitions *from* START_TAG
        self.transitions.data[:, self.tag_to_ix[START_TAG]] = -10000  # No transitions *to* START_TAG
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000   # No transitions *to* STOP_TAG
        self.transitions.data[self.tag_to_ix[STOP_TAG], :] = -10000   # No transitions *from* STOP_TAG


    def _get_lstm_features(self, sentences, lengths):
        """
        Pass input sentences through BiLSTM to get emission scores.
        """
        embeds = self.word_embeds(sentences)  # (batch_size, seq_len, embedding_dim)
        packed_embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)  # (batch_size, seq_len, hidden_dim)
        return self.hidden2tag(lstm_out)  # (batch_size, seq_len, tagset_size)

    def _forward_alg(self, feats, lengths):
        """
        Forward algorithm to compute the partition function.
        """
        batch_size, seq_len, tagset_size = feats.size()
        init_alphas = torch.full((batch_size, tagset_size), -10000., device=DEVICE)
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.0  # Start with START_TAG

        forward_var = init_alphas

        for i in range(seq_len):
            feat = feats[:, i, :]  # (batch_size, tagset_size)
            alphas_t = []

            for next_tag in range(tagset_size):
                emit_score = feat[:, next_tag].unsqueeze(1).expand(-1, tagset_size)  # (batch_size, tagset_size)
                trans_score = self.transitions[next_tag].unsqueeze(0).expand(batch_size, -1)  # (batch_size, tagset_size)
                next_tag_var = forward_var + trans_score + emit_score  # (batch_size, tagset_size)
                alphas_t.append(log_sum_exp(next_tag_var).squeeze(1))  # (batch_size,)

            forward_var = torch.stack(alphas_t, dim=1)  # (batch_size, tagset_size)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0)
        return log_sum_exp(terminal_var).squeeze(-1)  # (batch_size,)

    def _score_sentence(self, feats, tags, lengths):
        """
        Compute the score of a given tag sequence.
        """
        batch_size, seq_len, _ = feats.size()
        score = torch.zeros(batch_size, device=DEVICE)

        tags = torch.cat([
            torch.full((batch_size, 1), self.tag_to_ix[START_TAG], dtype=torch.long, device=DEVICE),
            tags
        ], dim=1)  # (batch_size, seq_len + 1)

        for i in range(seq_len):
            current_tag, next_tag = tags[:, i], tags[:, i + 1]
            emit_score = feats[torch.arange(batch_size), i, next_tag]  # Emission score
            trans_score = self.transitions[next_tag, current_tag]      # Transition score
            score += emit_score + trans_score

        last_tag_indices = lengths - 1
        last_tags = tags[torch.arange(batch_size), last_tag_indices + 1]
        score += self.transitions[self.tag_to_ix[STOP_TAG], last_tags]
        return score.view(-1)  # (batch_size,)

    def neg_log_likelihood(self, sentences, tags, lengths):
        """
        Calculate the negative log-likelihood loss.
        """
        feats = self._get_lstm_features(sentences, lengths)  # (batch_size, seq_len, tagset_size)
        forward_score = self._forward_alg(feats, lengths)     # Partition function
        gold_score = self._score_sentence(feats, tags, lengths)  # Gold sequence score
        return torch.mean(forward_score - gold_score)  # Average loss over batch

    def forward(self, sentences, lengths):
        feats = self._get_lstm_features(sentences, lengths)  # (batch_size, seq_len, tagset_size)
        batch_size, seq_len, _ = feats.size()
        predictions = []

        for i in range(batch_size):
            seq_len_i = lengths[i].item()
            forward_var = torch.full((self.tagset_size,), -10000., device=DEVICE)
            forward_var[self.tag_to_ix[START_TAG]] = 0.0  # Initialize start

            pred_tags = []
            for t in range(seq_len_i):
                emit_score = feats[i, t]  # (tagset_size,)

                # MASK OUT START_TAG DURING PREDICTION
                mask = torch.full((self.tagset_size,), -10000., device=DEVICE)
                mask[self.tag_to_ix[START_TAG]] = -10000.0  # Prevent predicting START

                scores = forward_var + emit_score + self.transitions[:, self.tag_to_ix[START_TAG]] + mask
                best_tag = torch.argmax(scores).item()
                pred_tags.append(best_tag)
                forward_var = scores

            predictions.append(pred_tags)

        return predictions
