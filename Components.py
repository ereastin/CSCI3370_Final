import torch
import torch.nn as nn

EPS = 1e-5  # this is the default for BatchNorm

# ---------------------------------------------------------------------------------
class Conv3(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, d=1, b=False, drop_p=0.0, res=False):
        super(Conv3, self).__init__()
        self.res = res
        self.net = nn.Sequential(
            nn.LazyConv3d(c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=b),
            nn.SELU(),
            # nn.LazyBatchNorm3d(eps=EPS),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        if self.res:
            return x + self.net(x)
        else:
            return self.net(x)

# ---------------------------------------------------------------------------------
class Conv2(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, d=1, b=False, drop_p=0.0, res=False):
        super(Conv2, self).__init__()
        self.res = res
        self.net = nn.Sequential(
            nn.LazyConv2d(c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=b),
            nn.SELU(),
            # nn.LazyBatchNorm2d(eps=EPS),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        if self.res:
            return x + self.net(x)
        else:
            return self.net(x)

# TODO: if wanting residual connections here need to make sure TConv is producing same shape
# btw if it is producing same shape is it same as Conv or no.? why not use Conv then?
# ---------------------------------------------------------------------------------
class TConv3(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, op=0, b=False, drop_p=0.0):
        super(TConv3, self).__init__()
        self.p = p
        self.net = nn.Sequential(
            nn.LazyConvTranspose3d(c_out, kernel_size=k, stride=s, padding=p, output_padding=op, bias=b),
            nn.SELU(),
            # nn.LazyBatchNorm3d(eps=EPS),
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
            nn.SELU(),
            # nn.LazyBatchNorm2d(eps=EPS),
            nn.Dropout2d(p=drop_p)
        )
        

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------------
class SelectionUnit2(nn.Module):
    def __init__(self, c_out):
        super(SelectionUnit2, self).__init__()
        self.net = nn.Sequential(
            nn.SELU(),
            nn.LazyConv2d(c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #NOTE: if wanting to see selection on these need to return output of net()
        return x * self.net(x)

# ---------------------------------------------------------------------------------
class SelectionUnit3(nn.Module):
    def __init__(self, c_out):
        super(SelectionUnit3, self).__init__()
        self.net = nn.Sequential(
            nn.SELU(),
            nn.LazyConv3d(c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)
