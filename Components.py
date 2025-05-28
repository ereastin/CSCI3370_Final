import torch
import torch.nn as nn

EPS = 1e-5  # this is the default for BatchNorm

# ---------------------------------------------------------------------------------
class Conv3(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, d=1, b=False, drop_p=0.0):
        super(Conv3, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv3d(c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=b),
            nn.ReLU(),
            nn.LazyBatchNorm3d(eps=EPS),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------------
class Conv2(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, b=False, drop_p=0.0):
        super(Conv2, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(c_out, kernel_size=k, stride=s, padding=p, bias=b),
            nn.ReLU(),
            nn.LazyBatchNorm2d(eps=EPS),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------------
class TConv3(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, op=0, b=False, drop_p=0.0):
        super(TConv3, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConvTranspose3d(c_out, kernel_size=k, stride=s, padding=p, output_padding=op, bias=b),
            nn.ReLU(),
            nn.LazyBatchNorm3d(eps=EPS),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------------
class TConv2(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, op=0, b=False, drop_p=0.0):
        super(TConv2, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConvTranspose2d(c_out, kernel_size=k, stride=s, padding=p, output_padding=op, bias=b),
            nn.ReLU(),
            nn.LazyBatchNorm2d(eps=EPS),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        return self.net(x)

