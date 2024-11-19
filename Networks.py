# -*- coding: utf-8 -*-
"""Networks.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vQB3zZi3959vzlrOxWIEh7SaN3S_KLSs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import xarray as xr
import os

class CNN(nn.Module):
  def __init__(self, in_dim, hidden_dim=64, depth=5):
    super(CNN, self).__init__()
    self.in_dim = in_dim  # number of channels in input -> needed?
    self.hidden_dim = hidden_dim
    blocks = []
    blocks.append(self.cnn_block(in_dim, hidden_dim))
    for i in range(depth - 2):
      blocks.append(self.cnn_block(hidden_dim, hidden_dim))
    blocks.append(self.cnn_block(hidden_dim, 1))  # final layer, 1 channel
    self.net = nn.Sequential(*blocks)

  def cnn_block(self, in_size, out_size):
    return nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_size),
        nn.ReLU()
    )

  def forward(self, x):
    out = self.net(x)
    return out

class UNetConvBlock(nn.Module):
  def __init__(self, in_size, out_size):
    super(UNetConvBlock, self).__init__()

    self.net = nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_size),
        nn.ReLU(),
        nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_size),
        nn.ReLU()
    )

  def forward(self, x):
    out = self.net(x)
    return out

class UNetDeconvBlock(nn.Module):
  def __init__(self, in_size, out_size):
    super(UNetDeconvBlock, self).__init__()

    self.upsmpl = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
    self.conv = UNetConvBlock(in_size, out_size)

  def forward(self, x_up, x_down):
    x_up = self.upsmpl(x_up)
    out = self.conv(torch.cat([x_up, x_down], dim=1))
    return out

def build_down_path(block, in_size, start_pwr, depth):
  down_path = nn.ModuleList()
  for i in range(depth):
    out_size = 2 ** (start_pwr + i)
    down_path.append(block(in_size, out_size))
    in_size = out_size
  return down_path

def build_up_path(block, in_size, depth):
  up_path = nn.ModuleList()
  for i in range(1, depth):
    out_size = in_size // 2
    up_path.append(block(in_size, out_size))
    in_size = out_size
  return up_path

class UNet(nn.Module):
  """
  would need to handle padding if using this on weird input grid sizes.?
  """
  def __init__(self, in_size, depth=5):
    super(UNet, self).__init__()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.final = nn.LazyConv2d(1, kernel_size=3, padding=1)

    start_pwr = 6  # 2 ** 6 channels after first conv
    self.down = build_down_path(UNetConvBlock, in_size, start_pwr, depth)

    start_pwr += depth - 1
    in_size = 2 ** start_pwr
    self.up = build_up_path(UNetDeconvBlock, in_size, depth)

  def forward(self, x):
    output_steps = []  # could make this clearer with a dictionary?
    for i, step in enumerate(self.down):
      x = step(x)
      if i < len(self.down) - 1:
        output_steps.append(x)
        x = self.pool(x)

    for i, step in enumerate(self.up):
      x = step(x, output_steps[-1 - i])

    out = self.final(x)
    return out

# test for correct dimensions and mem usage
test_in = torch.zeros((1, 64, 96, 96))
net = UNet(64)
net(test_in)

class Attention2D(nn.Module):
  def __init__(self, dwn_channels, up_channels, int_channels):
    """
    from Attention UNet paper..
    Zhang et al use BatchNorm and ReLU in defined conv steps.?
    g is deeper input (i.e. double features half HxW)
    x is downsampled input from 'left' side of UNet
    """
    super(Attention2D, self).__init__()
    self.W_x = nn.Conv2d(dwn_channels, int_channels, kernel_size=3, padding=1, stride=2, bias=True)  # reduce to dim_g
    self.W_g = nn.Conv2d(up_channels, int_channels, kernel_size=1, padding=0, stride=1, bias=True)
    self.psi = nn.Sequential(
        nn.Conv2d(int_channels, 1, kernel_size=1, padding=0, stride=1, bias=True),
        nn.Sigmoid(),
        nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
    )

  def forward(self, g, x):
    xx = self.W_x(x)  # (B, dwn_channels, H, W) -> (B, int_channels, H/2, W/2)
    gg = self.W_g(g)  # (B, up_channels, H/2, W/2) -> (B, int_channels, H/2, W/2)
    out = F.relu(xx + gg)
    psi = self.psi(out)
    res = psi * x  # hadamard prod (B, dwn_channels, H/2, W/2) -> cat to upsampled output
    return res

def build_attn_path(up_size, depth):
  """
  up_size will be channel depth (feature dim) of the deepest input i.e. up_channels
  -> build another insize that is half this to match 'x' input i.e. dwn_channels
  """
  attn_path = nn.ModuleList()
  for i in range(1, depth):
    dwn_size = up_size // 2
    attn_path.append(Attention2D(dwn_size, up_size, dwn_size))  # set int_size as dwn_size..?
    up_size //= 2

  return attn_path

class AttentionUNet(nn.Module):
  def __init__(self, in_size, depth=5):
    super(AttentionUNet, self).__init__()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.final = nn.LazyConv2d(1, kernel_size=3, padding=1)

    start_pwr = 6
    self.down = build_down_path(UNetConvBlock, in_size, start_pwr, depth)

    start_pwr += depth - 1
    in_size = 2 ** start_pwr
    self.up = build_up_path(UNetDeconvBlock, in_size, depth)

    self.attn = build_attn_path(in_size, depth)

  def forward(self, x):
    output_steps = []
    for i, step in enumerate(self.down):
      x = step(x)
      if i < len(self.down) - 1:
        output_steps.append(x)
        x = self.pool(x)

    for i, step in enumerate(self.up):
      att = self.attn[i]
      att_out = att(x, output_steps[-1 - i])
      x = step(x, att_out)

    out = self.final(x)
    return out

# test for correct dimensions and mem usage
test_in = torch.zeros((1, 64, 96, 96))
net = AttentionUNet(64)
net(test_in)

class InceptionBlock(nn.Module):
  def __init__(self, outb1, outb2a, outb2b, outb3a, outb3b, outb4b):
    # from D2L website
    super(InceptionBlock, self).__init__()
    self.b1 = nn.LazyConv2d(outb1, kernel_size=1)
    self.b2a = nn.LazyConv2d(outb2a, kernel_size=1)
    self.b2b = nn.LazyConv2d(outb2b, kernel_size=3, padding=1)
    self.b3a = nn.LazyConv2d(outb3a, kernel_size=1)
    self.b3b = nn.LazyConv2d(outb3b, kernel_size=5, padding=2)
    self.b4a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.b4b = nn.LazyConv2d(outb4b, kernel_size=1)

  def forward(self, x):
    x1 = F.relu(self.b1(x))
    x2 = F.relu(self.b2b(self.b2a(x)))
    x3 = F.relu(self.b3a(self.b3b(x)))
    x4 = F.relu(self.b4a(self.b4b(x)))
    return torch.cat([x1, x2, x3, x4], dim=1)  # cat along channel dim

class CAEEncoder(nn.Module):
  def __init__(self, in_size, out_size):
    super(CAEEncoder, self).__init__()
    # TODO: add Inception blocks -> need to sort out all those input params
    self.net = nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU()
    )

  def forward(self, x):
    out = self.net(x)
    return out

class CAEDecoder(nn.Module):
  def __init__(self, in_size, out_size):
    super(CAEDecoder, self).__init__()
    self.net = nn.Sequential(
        nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2),
        nn.LazyBatchNorm2d(),  # is batch norm an issue for physical meaning or does that matter here
        nn.ReLU()
    )

  def forward(self, x):
    out = self.net(x)
    return out

class CAE_LSTM(nn.Module):
  def __init__(self, in_size, lin_out1=2048, lin_out2=18432, hidden_size=256, depth=4):
    """
    in_size: number of channels of the input, e.g. 64 for 8 ERA5 vars at 8 pressure heights
    lin_out1: input feature dimension from first linear layer to LSTM
    lin_out2: output feature dimension -> unflatten to
    """
    super(CAE_LSTM, self).__init__()

    self.flat = nn.Flatten()
    unflat_shape = (512, 6, 6)
    self.unflat = nn.Unflatten(1, unflat_shape)  # should give (B, 128, 6, 6)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.linear1 = nn.LazyLinear(lin_out1)
    # how exactly should we handle this? want batch TO BE the sequence length.. right?
    self.lstm = nn.LSTM(lin_out1, hidden_size, bias=False, batch_first=True, dropout=0.0)  # TODO: define
    self.linear2 = nn.LazyLinear(lin_out2)
    self.final = nn.LazyConv2d(1, kernel_size=3, padding=1)
    start_pwr = 6  # first conv should be 128 channels
    self.down = build_down_path(CAEEncoder, in_size, start_pwr, depth)
    start_pwr += depth - 1
    in_size = 2 ** start_pwr
    self.up = build_up_path(CAEDecoder, in_size, depth + 1)

  def forward(self, x):
    # conv encoder
    for step in self.down:
      x = step(x)
      x = self.pool(x)

    # linear
    x = self.flat(x)  # (B, C * H * W)
    x = self.linear1(x)  # need size (B, L, input_size) for batch size B, sequence length L, input features input_size
    # lstm
    # need to rework sequence len stuff is this right..? should get 3dim tensor not 2
    x, _ = self.lstm(x)  # TODO: add attention! if using 1 layer probably don't need custom
    # linear
    x = self.linear2(x)
    x = self.unflat(x)

    # conv decoder
    for step in self.up:
      x = step(x)

    out = self.final(x)

    return out

test = torch.zeros((2, 64, 96, 96))
net = CAE_LSTM(64)
net(test)

