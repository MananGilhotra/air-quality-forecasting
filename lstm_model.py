# lstm_model.py
# Phase 2, step 3a: the LSTM network.
# An LSTM reads the 48 hours one by one and keeps a "memory" of what it has seen.
# After the last hour, a linear layer turns that memory into one number:
# the predicted change in PM2.5 over the next 6 hours.

import torch
import torch.nn as nn
import config


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=config.LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, 48 hours, n_features)
        lstm_out, _ = self.lstm(x)
        last_hour = lstm_out[:, -1, :]         # the memory after reading all 48 hours
        return self.output(last_hour).squeeze(-1)


if __name__ == "__main__":
    model = LSTMForecaster(n_features=16)
    fake_batch = torch.randn(8, config.SEQ_LEN, 16)      # 8 samples, 48 hours, 16 features
    print(model)
    print("Input shape :", tuple(fake_batch.shape))
    print("Output shape:", tuple(model(fake_batch).shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
