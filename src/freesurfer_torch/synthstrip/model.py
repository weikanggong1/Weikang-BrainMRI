"""SynthStrip signed-distance network with official checkpoint state keys."""

import numpy as np
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1, activation="leaky"):
        super().__init__()
        self.conv = getattr(nn, f"Conv{ndims}d")(
            in_channels, out_channels, 3, stride, 1
        )
        if activation == "leaky":
            self.activation = nn.LeakyReLU(0.2)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x) if self.activation is not None else x


class StripModel(nn.Module):
    """Official seven-level, signed-distance U-Net, with unchanged state keys."""

    def __init__(
        self,
        nb_features=16,
        nb_levels=7,
        feat_mult=2,
        max_features=64,
        nb_conv_per_level=2,
        max_pool=2,
        return_mask=False,
    ):
        super().__init__()
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError("must provide nb_levels when nb_features is an integer")
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError("cannot use nb_levels when nb_features is not an integer")

        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        self.pooling = [nn.MaxPool3d(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool]

        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(3, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(3, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            prev_nf += encoder_nfs[level]

        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(3, prev_nf, nf))
            prev_nf = nf
        self.remaining.append(ConvBlock(3, prev_nf, 2 if return_mask else 1, activation=None))
        if return_mask:
            self.remaining.append(nn.Softmax(dim=1))

    def forward(self, x):
        history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            history.append(x)
            x = self.pooling[level](x)
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            x = self.upsampling[level](x)
            x = torch.cat([x, history.pop()], dim=1)
        for conv in self.remaining:
            x = conv(x)
        return x
