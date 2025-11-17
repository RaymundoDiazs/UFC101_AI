import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):  # x: [B,T,in_dim]
        _, (h, _) = self.lstm(x)
        h_last = h[-1]  # [B, hidden]
        logits = self.fc(h_last)
        return logits
