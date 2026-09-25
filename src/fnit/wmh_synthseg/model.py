"""WMH-SynthSeg v10 3D U-Net with the original checkpoint parameter names.

Adapted from FreeSurfer 8.2.0-1 ``WMHSynthSeg/unet3d/{model,buildingblocks}.py``,
which derives from Adrian Wolny's MIT-licensed pytorch-3dunet. See
``THIRD_PARTY_NOTICES.md`` and ``licenses/pytorch-3dunet-MIT.txt``.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_groups):
        super().__init__()
        groups = 1 if in_channels < num_groups else num_groups
        self.add_module('groupnorm', nn.GroupNorm(groups, in_channels))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False))
        self.add_module('LeakyReLU', nn.LeakyReLU(inplace=True))


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, *, encoder, num_groups):
        super().__init__()
        middle = max(in_channels, out_channels // 2) if encoder else out_channels
        self.add_module('SingleConv1', _SingleConv(in_channels, middle, num_groups))
        self.add_module('SingleConv2', _SingleConv(middle, out_channels, num_groups))


class _Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, *, pooling, num_groups):
        super().__init__()
        self.pooling = nn.MaxPool3d(2) if pooling else None
        self.basic_module = _DoubleConv(in_channels, out_channels, encoder=True,
                                        num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)


class _InterpolateUpsampling(nn.Module):
    def forward(self, encoder_features, x):
        size = encoder_features.shape[2:]
        if size[0] * size[1] * size[2] * x.shape[1] < 2e9:
            return F.interpolate(x, size=size, mode='nearest')
        parts = [F.interpolate(x[:, i:i + 10], size=size, mode='nearest')
                 for i in range(0, x.shape[1], 10)]
        return torch.cat(parts, dim=1)


class _Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, *, num_groups):
        super().__init__()
        self.upsampling = _InterpolateUpsampling()
        self.basic_module = _DoubleConv(in_channels, out_channels, encoder=False,
                                        num_groups=num_groups)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features, x)
        return self.basic_module(torch.cat((encoder_features, x), dim=1))


class UNet3D(nn.Module):
    """The 1-input, 39-output, five-level WMH-SynthSeg v10 architecture."""

    def __init__(self, in_channels=1, out_channels=39, final_sigmoid=False, f_maps=64,
                 layer_order='gcl', num_groups=8, num_levels=5,
                 is_segmentation=False, conv_padding=1, is3d=True):
        super().__init__()
        config = (in_channels, out_channels, final_sigmoid, f_maps, layer_order,
                  num_groups, num_levels, is_segmentation, conv_padding, is3d)
        expected = (1, 39, False, 64, 'gcl', 8, 5, False, 1, True)
        if config != expected:
            raise ValueError('WMH-SynthSeg v10 requires the official UNet3D configuration')

        features = [f_maps * 2 ** level for level in range(num_levels)]
        self.encoders = nn.ModuleList(
            _Encoder(in_channels if i == 0 else features[i - 1], channels,
                     pooling=i > 0, num_groups=num_groups)
            for i, channels in enumerate(features)
        )
        reversed_features = list(reversed(features))
        self.decoders = nn.ModuleList(
            _Decoder(reversed_features[i] + reversed_features[i + 1],
                     reversed_features[i + 1], num_groups=num_groups)
            for i in range(num_levels - 1)
        )
        self.final_conv = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x):
        encoded = []
        for encoder in self.encoders:
            x = encoder(x)
            encoded.insert(0, x)
        for decoder, skip in zip(self.decoders, encoded[1:]):
            x = decoder(skip, x)
        return self.final_conv(x)
