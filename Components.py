import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5  # this is the default for BatchNorm

# ---------------------------------------------------------------------------------
class Attn(nn.Module):
    def __init__(self, c_int, c_out, depth, b=False, twod=False):
        super(Attn, self).__init__()
        if twod:
            self.squeeze = nn.Sequential(
                Conv2(c_int, k=1, s=1, p=0, b=b),
            )
        else:
            self.squeeze = nn.Sequential(
                Conv3(c_int, k=(depth, 1, 1), s=1, p=0, b=b),
            )

        self.layer_sig = nn.Sequential(
            Conv2(c_int, k=1, s=1, p=0, b=b), 
        )
        self.gate_sig = nn.Sequential(
            Conv2(c_int, k=1, s=1, p=0, b=b),
        )
        self.relu = nn.ReLU()
        self.psi = nn.Sequential(
            Conv2(1, k=1, s=1, p=0, b=b),
            nn.Sigmoid()
        )

    def forward(self, gate, layer):
        layer = self.squeeze(layer).squeeze(2)
        g = self.gate_sig(gate)
        l = self.layer_sig(layer)
        out = self.relu(l + g)
        return layer * self.psi(out)

# ---------------------------------------------------------------------------------
class Conv3(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, d=1, b=False, drop_p=0.0, res=False):
        super(Conv3, self).__init__()
        self.res = res
        self.net = nn.Sequential(
            nn.LazyConv3d(c_out, kernel_size=k, stride=s, padding=p, padding_mode='replicate', dilation=d, bias=b),
            nn.ReLU(),
            #nn.NORM?3d(eps=EPS, track_running_stats=False),
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
            nn.LazyConv2d(c_out, kernel_size=k, stride=s, padding=p, padding_mode='replicate', dilation=d, bias=b),
            nn.ReLU(),
            #nn.NORM?2d(eps=EPS, track_running_stats=False),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        if self.res:
            return x + self.net(x)
        else:
            return self.net(x)

# ---------------------------------------------------------------------------------
class TConv3(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, op=0, b=False, drop_p=0.0):
        super(TConv3, self).__init__()
        self.p = p
        self.net = nn.Sequential(
            nn.LazyConvTranspose3d(c_out, kernel_size=k, stride=s, padding=p, output_padding=op, bias=b),
            nn.ReLU(),
            #nn.NORM?3d(eps=EPS, track_running_stats=False),
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
            #nn.NORM?2d(eps=EPS, track_running_stats=False),
            nn.Dropout2d(p=drop_p)
        )
        

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------------
class Upsample2(nn.Module):
    def __init__(self, c_out, k=3, s=1, p=1, b=False, drop_p=0.0):
        super(Upsample2, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(c_out, kernel_size=k, stride=s, padding=p, padding_mode='replicate', bias=b),
            nn.ReLU(),
            # addition of below to correct upsampling artifacts?
            nn.LazyConv2d(c_out, kernel_size=k, stride=s, padding=p, padding_mode='replicate', bias=b),
            nn.ReLU(),
            nn.LazyConv2d(c_out, kernel_size=k, stride=s, padding=p, padding_mode='replicate', bias=b),
            nn.ReLU(),
            #nn.NORM?2d(eps=EPS, track_running_stats=False),
            nn.Dropout2d(p=drop_p)
        )
        

    def forward(self, x, size):
        up = F.interpolate(x, size=size, mode='nearest')
        return self.net(up)

# ---------------------------------------------------------------------------------
class SelectionUnit2(nn.Module):
    def __init__(self, c_out):
        super(SelectionUnit2, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
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
            nn.ReLU(),
            nn.LazyConv3d(c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)
