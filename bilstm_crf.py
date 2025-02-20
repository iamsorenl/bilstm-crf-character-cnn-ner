# ---------------------------------------------------
# BiLSTM-CRF Model for Named Entity Recognition (NER)
# Based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# ---------------------------------------------------

# ------------------------- Imports -------------------------
import torch
import torch.nn as nn  # For neural network layers (LSTM, Linear, Embedding)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # For handling variable-length sequences

# ------------------------- Helper Functions -------------------------

def argmax(vec):
    """
    Returns the index of the maximum value in the vector.
    Used in the Viterbi decoding to select the best tag.
    """
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    """
    Computes log-sum-exp in a numerically stable way.
    Used in the CRF forward algorithm to avoid underflow.
    """
    max_score = vec[0, argmax(vec)]
    # Broadcast max_score for numerical stability during exp calculation
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# ------------------------- BiLSTM-CRF Model -------------------------

class BiLSTM_CRF(nn.Module):
    def __init__(self, token_vocab_size, label_vocab, embedding_dim, hidden_dim):
        """
        Initialize the BiLSTM-CRF model components:
        - Embedding layer: Converts token indices into dense vectors.
        - BiLSTM: Captures contextual information from both directions.
        - Linear layer: Maps LSTM outputs to tag space.
        - CRF transition matrix: Scores transitions between tags.
        """
        super(BiLSTM_CRF, self).__init__()
        
        # Model hyperparameters and vocab sizes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = token_vocab_size
        self.label_vocab = label_vocab
        self.tagset_size = len(label_vocab)

        # Word embedding layer
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Bi-directional LSTM: hidden_dim // 2 per direction (forward & backward)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            bidirectional=True
        )

        # Linear layer: maps LSTM outputs to tag scores (emission scores)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # CRF Transition matrix: transition scores from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # Enforce constraints:
        # - No transition to START_TAG
        # - No transition from STOP_TAG
        self.transitions.data[self.label_vocab["<START>"], :] = -10000
        self.transitions.data[:, self.label_vocab["<STOP>"]] = -10000


    def init_hidden(self, batch_size, device=None):
        device = device or next(self.parameters()).device  # Use model's device
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
            torch.randn(2, batch_size, self.hidden_dim // 2, device=device)
        )


    def _get_lstm_features(self, sentences, lengths):
        """
        Extract LSTM features (emission scores) for each token in the sentence.
        Steps:
        - Embed the input tokens.
        - Pack the sequences to handle variable lengths.
        - Pass through BiLSTM.
        - Unpack and pass through the linear layer to get tag scores.
        """
        batch_size = sentences.size(0)
        self.hidden = self.init_hidden(batch_size, sentences.device)


        # Step 1: Embedding layer: (batch_size, seq_len, embedding_dim)
        embeds = self.word_embeds(sentences)

        # Step 2: Pack sequences (handles variable lengths efficiently)
        packed_embeds = pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Step 3: Pass through BiLSTM
        packed_lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)

        # Step 4: Unpack sequences to get padded output
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        # Step 5: Linear layer to convert LSTM outputs to tag scores
        lstm_feats = self.hidden2tag(lstm_out)  # Shape: (batch_size, seq_len, tagset_size)

        return lstm_feats

    def _forward_alg(self, feats):
        """
        Forward algorithm to compute the log-sum-exp of all possible tag sequences.
        This calculates the partition function Z(x) for CRF normalization.
        Steps:
        - Initialize forward variables with -inf except for START_TAG.
        - Iterate over the sequence:
            - Calculate emission + transition scores.
            - Perform log-sum-exp to accumulate scores.
        - Add transition to STOP_TAG and perform final log-sum-exp.
        """
        batch_size, seq_len, tagset_size = feats.size()

        # Initialize forward variables with -10000 (log(0)) except START_TAG = 0
        init_alphas = torch.full((batch_size, tagset_size), -10000., device=feats.device)
        init_alphas[:, self.label_vocab["<START>"]] = 0.

        forward_var = init_alphas  # (batch_size, tagset_size)

        # Iterate through the sequence (time steps)
        for i in range(seq_len):
            emit_scores = feats[:, i].unsqueeze(2)  # (batch_size, tagset_size, 1)
            trans_scores = self.transitions.unsqueeze(0)  # (1, tagset_size, tagset_size)

            # Calculate score for all tag transitions and emissions
            scores = forward_var.unsqueeze(1) + emit_scores + trans_scores  # (batch_size, tagset_size, tagset_size)

            # Perform log-sum-exp across the previous tags
            forward_var = torch.logsumexp(scores, dim=2)  # (batch_size, tagset_size)

        # Add transitions to STOP_TAG
        terminal_var = forward_var + self.transitions[self.label_vocab["<STOP>"]]
        
        # Final log-sum-exp gives the partition function for the batch
        return torch.logsumexp(terminal_var, dim=1)  # (batch_size,)

    def _score_sentence(self, feats, tags, lengths):
        """
        Computes the score for the provided tag sequence.
        Steps:
        - Add the emission and transition scores for each token-tag pair.
        - Include the transition to the STOP_TAG.
        """
        batch_size, seq_len, _ = feats.size()
        score = torch.zeros(batch_size, device=feats.device)
        
        # Add START_TAG at the beginning of each sequence
        start_tags = torch.full((batch_size, 1), self.label_vocab["<START>"], dtype=torch.long, device=feats.device)
        tags = torch.cat([start_tags, tags], dim=1)  # (batch_size, seq_len + 1)

        for i in range(seq_len):
            current_tag = tags[:, i]
            next_tag = tags[:, i + 1]
            emit_score = feats[torch.arange(batch_size), i, next_tag]
            trans_score = self.transitions[next_tag, current_tag]
            score += emit_score + trans_score

        # Add transition to STOP_TAG
        last_tag = tags[torch.arange(batch_size), lengths]
        score += self.transitions[self.label_vocab["<STOP>"], last_tag]
        return score

    def neg_log_likelihood(self, sentences, tags, lengths):
        """
        Computes the negative log-likelihood loss:
        Loss = log_sum_exp(forward scores) - score of true tag sequence
        """
        feats = self._get_lstm_features(sentences, lengths)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags, lengths)
        return torch.mean(forward_score - gold_score)  # Return average batch loss
    
    def _viterbi_decode(self, feats):
        """
        Performs Viterbi decoding to find the best tag sequence.
        """
        batch_size, seq_len, tagset_size = feats.size()
        backpointers = []

        # Initialize viterbi variables
        init_vvars = torch.full((batch_size, tagset_size), -10000., device=feats.device)
        init_vvars[:, self.label_vocab["<START>"]] = 0
        forward_var = init_vvars

        for i in range(seq_len):
            bptrs_t = []  # To store backpointers at time step i
            viterbivars_t = []  # To store viterbi variables at time step i

            for next_tag in range(tagset_size):
                # Scores for transitioning from all previous tags to `next_tag`
                next_tag_var = forward_var + self.transitions[next_tag].unsqueeze(0)
                best_tag_id = torch.argmax(next_tag_var, dim=1)  # (batch_size,)
                bptrs_t.append(best_tag_id)

                # Add emission score for current time step
                best_var = next_tag_var[torch.arange(batch_size), best_tag_id] + feats[:, i, next_tag]
                viterbivars_t.append(best_var)

            forward_var = torch.stack(viterbivars_t, dim=1)  # (batch_size, tagset_size)
            backpointers.append(torch.stack(bptrs_t, dim=1))  # (batch_size, tagset_size)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.label_vocab["<STOP>"]].unsqueeze(0)
        best_tag_id = torch.argmax(terminal_var, dim=1)  # (batch_size,)

        # Backtrace best path for each sequence in batch
        best_paths = []
        for batch_idx in range(batch_size):
            best_path = [best_tag_id[batch_idx].item()]
            for bptrs_t in reversed(backpointers):
                best_tag = bptrs_t[batch_idx, best_path[-1]]
                best_path.append(best_tag.item())

            # Remove the START tag and reverse
            best_paths.append(best_path[:-1][::-1])

        return terminal_var, best_paths
    
    def forward(self, sentences, lengths):
        """
        Forward pass:
        - Extract LSTM features.
        - Apply Viterbi decoding.
        - Return the score and predicted tags.
        """
        lstm_feats = self._get_lstm_features(sentences, lengths)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
