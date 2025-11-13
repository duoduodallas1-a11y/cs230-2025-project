import torch
import torch.nn as nn


# ---------- RNN (LSTM) ----------
class RNNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        _, (h, _) = self.lstm(x)      # h: (num_layers, B, H)
        out = self.fc(h[-1])          # last layer's hidden state
        return out                    # logits (no softmax here)


# ---------- TCN ----------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        return self.relu(out)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.block1 = TemporalBlock(input_dim, 64, kernel_size=3, dilation=1)
        self.block2 = TemporalBlock(64, 64, kernel_size=3, dilation=2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.block2(self.block1(x))   # (B, 64, T)
        h_pool = h.mean(dim=2)            # global average pooling over time
        return self.fc(h_pool)            # logits


# ---------- Transformer Encoder ----------
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        nheads: int = 4,
        nlayers: int = 2,
        num_classes: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        h = self.input_proj(x)      # (B, T, H)
        enc = self.encoder(h)       # (B, T, H)
        h_pool = enc.mean(dim=1)    # global average pooling
        return self.fc(h_pool)      # logits
