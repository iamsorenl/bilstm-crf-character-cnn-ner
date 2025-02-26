# ---------------------------------------------------
# BiLSTM-CRF Model for Named Entity Recognition (NER) with Character-level CNN
# Based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# ---------------------------------------------------

# ------------------------- Imports -------------------------
import torch
import torch.nn as nn  # For neural network layers (LSTM, Linear, Embedding)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # For handling variable-length sequences

# ------------------------- BiLSTM-CRF with Character CNN -------------------------
class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        token_vocab_size,
        char_vocab_size,
        label_vocab,
        embedding_dim,
        char_embedding_dim,
        char_out_dim,
        hidden_dim,
        char_kernel_size=3
    ):
        super(BiLSTM_CRF, self).__init__()

        # -------- Word Embeddings --------
        self.word_embeds = nn.Embedding(token_vocab_size, embedding_dim)

        '''
        # -------- Character Embeddings --------
        self.char_embeds = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_out_dim,
            kernel_size=char_kernel_size,
            padding=char_kernel_size // 2
        )
        '''

        # -------- LSTM Input Dimension --------
        #self.lstm_input_dim = embedding_dim + char_out_dim
        self.lstm_input_dim = embedding_dim

        # -------- BiLSTM Layer --------
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True
        )

        # -------- Linear Layer (Emission Scores) --------
        self.hidden2tag = nn.Linear(hidden_dim, len(label_vocab))

        # -------- CRF Transition Matrix --------
        self.transitions = nn.Parameter(torch.randn(len(label_vocab), len(label_vocab)))
        self.transitions.data[label_vocab["<START>"]][:] = -10000  # No transition to START_TAG
        self.transitions.data[:, label_vocab["<STOP>"]] = -10000   # No transition from STOP_TAG

        self.label_vocab = label_vocab

    def _get_char_features(self, char_sequences):
        """
        Extract character-level CNN features.
        char_sequences: Tensor (batch_size, seq_len, max_word_length)
        """
        batch_size, seq_len, max_word_len = char_sequences.size()

        # Step 1: Embed characters -> (batch_size, seq_len, max_word_len, char_embedding_dim)
        chars_embedded = self.char_embeds(char_sequences)

        # Step 2: Reshape for CNN input -> (batch_size * seq_len, char_embedding_dim, max_word_len)
        chars_embedded = chars_embedded.view(-1, max_word_len, chars_embedded.size(-1)).transpose(1, 2)

        # Step 3: Apply CNN + Max Pooling
        char_cnn_out = self.char_cnn(chars_embedded)  # (batch_size * seq_len, char_out_dim, max_word_len)
        char_features = torch.max(char_cnn_out, dim=2)[0]  # (batch_size * seq_len, char_out_dim)

        # Step 4: Reshape back -> (batch_size, seq_len, char_out_dim)
        return char_features.view(batch_size, seq_len, -1)

    def _get_lstm_features(self, sentences, char_sequences, lengths):
        """
        Combines word and character embeddings, passes them through BiLSTM.
        """
        word_embeds = self.word_embeds(sentences)  # (batch_size, seq_len, embedding_dim)
        #char_features = self._get_char_features(char_sequences)  # (batch_size, seq_len, char_out_dim)

        # Concatenate embeddings -> (batch_size, seq_len, embedding_dim + char_out_dim)
        #embeddings = torch.cat([word_embeds, char_features], dim=2)
        embeddings = word_embeds

        # Pack and pass through BiLSTM
        packed_embeds = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        # Linear layer for emission scores -> (batch_size, seq_len, tagset_size)
        return self.hidden2tag(lstm_out)

    def _forward_alg(self, feats):
        """
        Computes the log-sum-exp of all possible tag sequences.
        Used to calculate the partition function for CRF normalization.
        """
        batch_size, seq_len, tagset_size = feats.size()

        init_alphas = torch.full((batch_size, tagset_size), -10000., device=feats.device)
        init_alphas[:, self.label_vocab["<START>"]] = 0.

        forward_var = init_alphas

        for i in range(seq_len):
            emit_scores = feats[:, i].unsqueeze(2)  # (batch_size, tagset_size, 1)
            trans_scores = self.transitions.unsqueeze(0)  # (1, tagset_size, tagset_size)

            scores = forward_var.unsqueeze(1) + emit_scores + trans_scores  # (batch_size, tagset_size, tagset_size)
            forward_var = torch.logsumexp(scores, dim=2)  # (batch_size, tagset_size)

        terminal_var = forward_var + self.transitions[self.label_vocab["<STOP>"]].unsqueeze(0)
        return torch.logsumexp(terminal_var, dim=1)  # (batch_size,)

    def _score_sentence(self, feats, tags, lengths):
        """
        Computes the score for the given tag sequence.
        """
        batch_size, seq_len, _ = feats.size()
        score = torch.zeros(batch_size, device=feats.device)

        start_tags = torch.full((batch_size, 1), self.label_vocab["<START>"], dtype=torch.long, device=feats.device)
        tags = torch.cat([start_tags, tags], dim=1)  # (batch_size, seq_len + 1)

        for i in range(seq_len):
            current_tag = tags[:, i]
            next_tag = tags[:, i + 1]
            emit_score = feats[torch.arange(batch_size), i, next_tag]
            trans_score = self.transitions[next_tag, current_tag]
            score += emit_score + trans_score

        last_tag = tags[torch.arange(batch_size), lengths]
        score += self.transitions[self.label_vocab["<STOP>"], last_tag]
        return score

    def neg_log_likelihood(self, sentences, char_sequences, tags, lengths):
        """
        Computes the negative log-likelihood loss.
        Loss = log_sum_exp(forward scores) - score of true tag sequence
        """
        feats = self._get_lstm_features(sentences, char_sequences, lengths)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags, lengths)
        return torch.mean(forward_score - gold_score)

    def _viterbi_decode(self, feats):
        """
        Performs Viterbi decoding to find the best tag sequence.
        """
        batch_size, seq_len, tagset_size = feats.size()
        backpointers = []

        init_vvars = torch.full((batch_size, tagset_size), -10000., device=feats.device)
        init_vvars[:, self.label_vocab["<START>"]] = 0.
        forward_var = init_vvars

        for i in range(seq_len):
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag].unsqueeze(0)
                best_tag_id = torch.argmax(next_tag_var, dim=1)
                bptrs_t.append(best_tag_id)

                best_var = next_tag_var[torch.arange(batch_size), best_tag_id] + feats[:, i, next_tag]
                viterbivars_t.append(best_var)

            forward_var = torch.stack(viterbivars_t, dim=1)
            backpointers.append(torch.stack(bptrs_t, dim=1))

        terminal_var = forward_var + self.transitions[self.label_vocab["<STOP>"]].unsqueeze(0)
        best_tag_id = torch.argmax(terminal_var, dim=1)

        best_paths = []
        for batch_idx in range(batch_size):
            best_path = [best_tag_id[batch_idx].item()]
            for bptrs_t in reversed(backpointers):
                best_tag = bptrs_t[batch_idx, best_path[-1]]
                best_path.append(best_tag.item())
            best_paths.append(best_path[:-1][::-1])

        return terminal_var, best_paths

    def forward(self, sentences, char_sequences, lengths):
        """
        Forward pass:
        - Extract LSTM features.
        - Apply Viterbi decoding.
        - Return the score and predicted tags.
        """
        feats = self._get_lstm_features(sentences, char_sequences, lengths)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
