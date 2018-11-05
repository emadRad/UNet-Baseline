import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def conv(in_channels, out_channels, kernel_size, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def up_conv(in_channels, out_channels, kernel_size, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2)

    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv(in_channels, out_channels, kernel_size=1))
