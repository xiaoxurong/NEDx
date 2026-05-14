import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import Emb


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)

        x = self.avg(x)
        return x



class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    # use moving average to smoother the time series
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.use_norm = configs.use_norm

        self.emb = Emb(configs.seq_len, configs.d_model)

        self.encoder_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for _ in range(configs.e_layers)
        ])

        self.classifier = nn.Linear(configs.d_model, 1)

    def forward(self, x):
        # x: (B, C, T)
        if self.use_norm:
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

        x = x.permute(0, 2, 1)   # (B, T, C)
        x = self.emb(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)        # global pooling
        logits = self.classifier(x)
        return logits

class Encoder(nn.Module):

    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 2, 1)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)

        return y_2