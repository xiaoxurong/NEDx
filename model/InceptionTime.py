"""
model/InceptionTime.py
InceptionTime: Fawaz et al., 2020 (https://arxiv.org/abs/1909.04939)
Multi-scale 1D CNN with residual connections for time-series classification.
Does not need seq_len — uses AdaptiveAvgPool1d for the final pooling step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _InceptionBlock(nn.Module):
    """Single Inception module: bottleneck → 3 parallel scales + maxpool branch."""
    def __init__(self, in_channels: int, nb_filters: int = 32, bottleneck_size: int = 32):
        super().__init__()
        self.use_bottleneck = in_channels > bottleneck_size
        bn_ch = bottleneck_size if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, 1, bias=False)

        self.conv_small  = nn.Conv1d(bn_ch, nb_filters, kernel_size=9,  padding=4,  bias=False)
        self.conv_medium = nn.Conv1d(bn_ch, nb_filters, kernel_size=19, padding=9,  bias=False)
        self.conv_large  = nn.Conv1d(bn_ch, nb_filters, kernel_size=39, padding=19, bias=False)
        self.maxpool     = nn.MaxPool1d(3, stride=1, padding=1)
        self.mp_conv     = nn.Conv1d(in_channels, nb_filters, 1, bias=False)
        self.bn          = nn.BatchNorm1d(nb_filters * 4)
        self.act         = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x) if self.use_bottleneck else x
        branches = [
            self.conv_small(z), self.conv_medium(z), self.conv_large(z),
            self.mp_conv(self.maxpool(x)),
        ]
        return self.act(self.bn(torch.cat(branches, dim=1)))


class _ResidualGroup(nn.Module):
    """Three InceptionBlocks with a residual shortcut around the group."""
    def __init__(self, in_channels: int, nb_filters: int = 32):
        super().__init__()
        nb_ch_out = nb_filters * 4
        self.block1 = _InceptionBlock(in_channels, nb_filters)
        self.block2 = _InceptionBlock(nb_ch_out,   nb_filters)
        self.block3 = _InceptionBlock(nb_ch_out,   nb_filters)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, nb_ch_out, 1, bias=False),
            nn.BatchNorm1d(nb_ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block3(self.block2(self.block1(x)))
        return F.relu(out + self.shortcut(x))


class Model(nn.Module):
    """
    Args (read from args):
        args.in_channels  — number of input channels (default 2: FHR + TOCO)
        args.d_model      — nb_filters per inception branch (output = d_model*4 per block)
        args.e_layers     — number of residual groups (each = 3 inception blocks)
        args.num_classes  — 2 for binary (outputs (B,1)), >2 for multi-class
    """
    def __init__(self, args):
        super().__init__()
        in_channels = getattr(args, 'in_channels', 2)
        nb_filters  = getattr(args, 'd_model',     32)
        num_groups  = getattr(args, 'e_layers',    2)
        out_dim     = 1 if args.num_classes == 2 else args.num_classes

        self.in_channels = in_channels
        nb_ch_out = nb_filters * 4
        groups = []
        ch = in_channels
        for _ in range(num_groups):
            groups.append(_ResidualGroup(ch, nb_filters))
            ch = nb_ch_out

        self.network    = nn.Sequential(*groups)
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(nb_ch_out, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise to (B, C, T) — data_provider returns (B, T, C)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        # Replace NaN (CTG signal dropouts) with 0
        x = torch.nan_to_num(x, nan=0.0)
        x = self.network(x)            # (B, nb_ch_out, T)
        x = self.gap(x).squeeze(-1)    # (B, nb_ch_out)
        return self.classifier(x)      # (B, out_dim)