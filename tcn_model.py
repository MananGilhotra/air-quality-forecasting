# tcn_model.py
# Phase 2, step 3b: the temporal CNN (TCN).
# Instead of reading hour by hour, a TCN slides small filters over the 48 hours.
# Each layer skips more hours (dilation 1, 2, 4, 8), so deeper layers see further back.
# The filters are "causal": the output for an hour only uses that hour and earlier ones.

import torch
import torch.nn as nn
import config


class TCNForecaster(nn.Module):
    def __init__(self, n_features, channels=config.TCN_CHANNELS, kernel_size=3,
                 dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.pads = []
        in_channels = n_features
        for dilation in dilations:
            pad = (kernel_size - 1) * dilation
            self.convs.append(nn.Conv1d(in_channels, channels, kernel_size,
                                        padding=pad, dilation=dilation))
            self.pads.append(pad)
            in_channels = channels
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(channels, 1)
        # How many hours back the last layer can see: 1 + 2 * (1 + 2 + 4 + 8) = 31
        self.history_used = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, x):
        # x shape: (batch, 48 hours, n_features)
        # Older hours can't reach the output anyway, so we drop them to train faster
        x = x[:, -(self.history_used + 1):, :]
        # Conv1d wants (batch, n_features, hours)
        x = x.transpose(1, 2)
        for conv, pad in zip(self.convs, self.pads):
            x = conv(x)[:, :, :-pad]           # cut the extra right side, so no future hour is used
            x = self.dropout(self.relu(x))
        last_hour = x[:, :, -1]                # the features at the most recent hour
        return self.output(last_hour).squeeze(-1)


if __name__ == "__main__":
    model = TCNForecaster(n_features=16)
    fake_batch = torch.randn(8, config.SEQ_LEN, 16)      # 8 samples, 48 hours, 16 features
    print(model)
    print("Input shape :", tuple(fake_batch.shape))
    print("Output shape:", tuple(model(fake_batch).shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
    print("Hours of history the last layer can see:", model.history_used)
