# dl_experiment_models.py
# Extra network designs we tried while looking for a more accurate model.
# Every class takes a batch of windows (batch, 48 hours, features)
# and returns one number per window: the predicted 6-hour change in PM2.5.

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """Like the LSTM, but with a simpler memory cell (fewer weights)."""

    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.output(gru_out[:, -1, :]).squeeze(-1)


class BiLSTMForecaster(nn.Module):
    """Reads the 48-hour window forwards AND backwards, then joins both summaries.
    Both directions only see past hours, because the window ends at the forecast hour."""

    def __init__(self, n_features, hidden_size=48):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        _, (last_hidden, _) = self.lstm(x)          # last_hidden: (2 directions, batch, hidden)
        both = torch.cat([last_hidden[0], last_hidden[1]], dim=1)
        return self.output(both).squeeze(-1)


class CNNLSTMForecaster(nn.Module):
    """A causal convolution first finds short 3-hour patterns, then an LSTM reads them in order."""

    def __init__(self, n_features, channels=32, hidden_size=64):
        super().__init__()
        self.conv = nn.Conv1d(n_features, channels, kernel_size=3, padding=2)
        self.lstm = nn.LSTM(input_size=channels, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2))[:, :, :-2]   # drop the 2 padded steps on the right: causal
        z = torch.relu(z).transpose(1, 2)
        lstm_out, _ = self.lstm(z)
        return self.output(lstm_out[:, -1, :]).squeeze(-1)


class AttentionLSTMForecaster(nn.Module):
    """An LSTM plus attention: the network gives every hour a weight
    and builds a weighted summary of the hours it finds most useful."""

    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.score = nn.Linear(hidden_size, 1)         # one score per hour
        self.output = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                     # (batch, 48, hidden)
        weights = torch.softmax(self.score(lstm_out), dim=1)   # the 48 weights add up to 1
        summary = (weights * lstm_out).sum(dim=1)
        both = torch.cat([summary, lstm_out[:, -1, :]], dim=1)
        return self.output(both).squeeze(-1)


class TransformerForecaster(nn.Module):
    """A small Transformer encoder: self-attention lets every hour look at every other hour."""

    def __init__(self, n_features, d_model=64, seq_len=48):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))   # learned "which hour" signal
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.input(x) + self.position[:, -x.shape[1]:, :]
        z = self.encoder(z)
        return self.output(z[:, -1, :]).squeeze(-1)
