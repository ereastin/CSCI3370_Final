import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import sys
sys.path.append('/home/eastinev/AI/')

from partialconv/models/partialconv3d import PartialConv3d

# TODO: partial conv is interesting idea.. but how does it work in UNet form..
#  e.g. how do you handle mask updates? take it all the way through?
#  then what happens at the bottom?

# ---------------------------------------------------------------------------------
def main():
    # Att U-Net paper has D = 96.. there are 38 pressure levels in ERA5 we could use them all
    net = IRNv4_3DUNet(7, depth=42, Na=5, Nb=10, Nc=5, bias=False, drop_p=0.0)  # 5 vars at 8 levels
    s = (64, 7, 42, 80, 144)  # one week ~24GB fwd/back pass, about same as before for one month
    # ~28GB if doing 5, 10, 5 for # of A, B, C layers
    summary(net, input_size=s)
    # TODO: handle 'base' better.?
    # TODO: handle padding/output padding stuff better! this will only work for known input shape
    #  need to somehow figure out even vs. odd shapes how to handle padding

def select_pad(shape_in, shape_out):
    D_in, H_in, W_in = shape_in[2:]
    D_out, H_out, W_out = shape_out[2:] 

def calc_shape_out(shape_in, p, d, k, s):
    return ((shape_in + 2 * p - d * (k - 1) - 1) / s + 1).__floor__()

# ---------------------------------------------------------------------------------
class S1(nn.Module):
  def __init__(self, in_size, base=64, b=False):
    super(S1, self).__init__()

    self.stem_a = nn.Sequential(
        PartialConv3d(in_size, base, kernel_size=3, stride=(1, 2, 2), padding=1, bias=b), nn.ReLU(),
        PartialConv3d(base, base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        PartialConv3d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(2 * base, eps=1e-2)
    )

  def forward(self, x):
    return self.stem_a(x)

# ---------------------------------------------------------------------------------
class S2(nn.Module):
  def __init__(self, in_size, base=64, b=False):
    super(S2, self).__init__()
    self.stem_b1 = nn.Sequential(
        nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
    )
    self.stem_b2 = nn.Sequential(
        PartialConv3d(2 * base, 3 * base, kernel_size=3, stride=(1, 2, 2), padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=1e-2)
    )

  def forward(self, x):
    return torch.cat([self.stem_b1(x), self.stem_b2(x)], dim=1)

# ---------------------------------------------------------------------------------
class S3(nn.Module):
  def __init__(self, in_size, base=64, b=False):
    super(S3, self).__init__()
    self.stem_c1 = nn.Sequential(
        PartialConv3d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(2 * base, 3 * base, kernel_size=3, stride=1, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=1e-2)
    )
    self.stem_c2 = nn.Sequential(
        PartialConv3d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        # TODO: check these, thoughts on >1 k_size for depth dim?
        PartialConv3d(2 * base, 2 * base, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0), bias=b), nn.ReLU(),
        PartialConv3d(2 * base, 2 * base, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3), bias=b), nn.ReLU(),
        PartialConv3d(2 * base, 3 * base, kernel_size=3, stride=1, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=1e-2)
    )

  def forward(self, x):
    return torch.cat([self.stem_c1(x), self.stem_c2(x)], dim=1)

# ---------------------------------------------------------------------------------
class S4(nn.Module):
  def __init__(self, in_size, b=False):
    super(S4, self).__init__()
    self.stem_d1 = nn.Sequential(
        PartialConv3d(in_size, in_size, kernel_size=3, stride=2, padding=(1, 1, 1), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size, eps=1e-2)
    )
    self.stem_d2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

  def forward(self, x):
    return F.relu(torch.cat([self.stem_d1(x), self.stem_d2(x)], dim=1))

# ---------------------------------------------------------------------------------
class A(nn.Module):
  def __init__(self, in_size, base=64, b=False):
    super(A, self).__init__()

    self.b1 = nn.Sequential(
        PartialConv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        PartialConv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(base, base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        PartialConv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(base, 3 * base // 2, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        PartialConv3d(3 * base // 2, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(2 * base, eps=1e-2)
    )
    self.comb = PartialConv3d(4 * base, in_size, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.ReLU()

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)  # filter cat right?
    x_conv = 0.1 * self.comb(x_conv) # linear activation scaling for stability

    return self.final(x + x_conv)  # residual connection

# ---------------------------------------------------------------------------------
class redA(nn.Module):
  def __init__(self, in_size, b=False, drop_p=0.0):
    super(redA, self).__init__()
    self.b1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=(1, 0, 0))
    self.b2 = nn.Sequential(
        PartialConv3d(in_size, in_size, kernel_size=3, stride=2, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        PartialConv3d(in_size, 3 * in_size // 2, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(3 * in_size // 2, 3 * in_size // 2, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        PartialConv3d(3 * in_size // 2, in_size, kernel_size=3, stride=2, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size, eps=1e-2)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))

# ---------------------------------------------------------------------------------
class B(nn.Module):
  def __init__(self, in_size, b=False):
    super(B, self).__init__()
    self.b1 = nn.Sequential(
        PartialConv3d(in_size, in_size // 8, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 8, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        PartialConv3d(in_size, in_size // 10, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(in_size // 10, in_size // 9, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0), bias=b), nn.ReLU(),
        PartialConv3d(in_size // 9, in_size // 8, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 8, eps=1e-2)
    )
    self.comb = PartialConv3d(in_size // 4, in_size, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.ReLU()

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = 0.1 * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class redB(nn.Module):
  def __init__(self, in_size, b=False, drop_p=0.0):
    super(redB, self).__init__()
    self.b1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=(1, 0, 0))
    self.b2 = nn.Sequential(
        PartialConv3d(in_size, 2 * in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(2 * in_size // 12, 5 * in_size // 12, kernel_size=3, stride=2, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(5 * in_size // 12, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        PartialConv3d(in_size, 2 * in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(2 * in_size // 12, 3 * in_size // 12, kernel_size=3, stride=2, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * in_size // 12, eps=1e-2)
    )
    self.b4 = nn.Sequential(
        PartialConv3d(in_size, 2 * in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(2 * in_size // 12, 3 * in_size // 12, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        PartialConv3d(3 * in_size // 12, 4 * in_size // 12, kernel_size=3, stride=2, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(4 * in_size // 12, eps=1e-2)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))

# ---------------------------------------------------------------------------------
class C(nn.Module):
  def __init__(self, in_size, b=False, drop_p=0.0):
    super(C, self).__init__()
    self.b1 = nn.Sequential(
        PartialConv3d(in_size, in_size // 10, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 10, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        PartialConv3d(in_size, in_size // 10, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        PartialConv3d(in_size // 10, in_size // 9, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=b), nn.ReLU(),
        PartialConv3d(in_size // 9, in_size // 8, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 8, eps=1e-2)
    )
    cat_size = in_size // 8 + in_size // 10
    self.comb = PartialConv3d(cat_size, in_size, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = 0.1 * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class revLB(nn.Module):
  def __init__(self, in_size, out_size, b=False, drop_p=0.0):
    super(revLB, self).__init__()
    self.b1 = nn.Sequential(
        nn.ConvTranspose3d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 12, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose3d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.ConvTranspose3d(in_size // 12, in_size // 6, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 6, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.ConvTranspose3d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.ConvTranspose3d(in_size //12, in_size // 8, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.ConvTranspose3d(in_size // 8, in_size // 4, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 4, eps=1e-2)
    )
    self.final = nn.Dropout3d(p=drop_p)

  def forward(self, x, cat_in):
    comb = torch.cat([x, cat_in], dim=1)
    return self.final(torch.cat([self.b1(comb), self.b2(comb), self.b3(comb)], dim=1))

# ---------------------------------------------------------------------------------
class revLA(nn.Module):
  def __init__(self, in_size, out_size, b=False, drop_p=0.0):
    super(revLA, self).__init__()
    self.b1 = nn.Sequential(
        nn.ConvTranspose3d(in_size, in_size // 6, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 6, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose3d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.ConvTranspose3d(in_size // 12, in_size // 8, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.ConvTranspose3d(in_size // 8, in_size // 3, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 3, eps=1e-2)
    )
    self.final = nn.Dropout3d(p=drop_p)

  def forward(self, x, cat_in):
    comb = torch.cat([x, cat_in], dim=1)
    return self.final(torch.cat([self.b1(comb), self.b2(comb)], dim=1))

# ---------------------------------------------------------------------------------
class revRL(nn.Module):
  def __init__(self, in_size, out_size, op=(0, 1, 1), b=False, drop_p=0.0):
    super(revRL, self).__init__()
    # TODO: worth breaking this up like A/B reduction ones? 
    ratio = in_size // out_size
    hid = 2 * ratio
    self.b1 = nn.Sequential(
        nn.ConvTranspose3d(in_size, in_size // 24, kernel_size=3, stride=2, padding=(1, 0, 0), output_padding=op, bias=b), nn.ReLU(),
        nn.ConvTranspose3d(in_size // 24, in_size // hid, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // hid, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose3d(in_size, in_size // 24, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.ConvTranspose3d(in_size // 24, in_size // hid, kernel_size=3, stride=2, padding=(1, 0, 0), output_padding=op, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // hid, eps=1e-2)
    ) 
    self.final = nn.Dropout3d(p=drop_p)

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x)], dim=1))

# ---------------------------------------------------------------------------------
class revS(nn.Module):
  def __init__(self, in_size, out_size, stride=2, p=(1, 1, 1), op=(1, 1, 1), b=False):
    super(revS, self).__init__()
    self.ct = nn.Sequential(
        nn.ConvTranspose3d(in_size, out_size, kernel_size=3, stride=stride, padding=p, output_padding=op, bias=b), nn.ReLU(),
        nn.BatchNorm3d(out_size, eps=1e-2)
    )

  def forward(self, x):
    return self.ct(x)

# ---------------------------------------------------------------------------------
class revScat(nn.Module):
  def __init__(self, in_size, out_size, stride=1, p=(1, 0, 0), op=0, b=False):
    super(revScat, self).__init__()
    self.ct = nn.Sequential(
        nn.ConvTranspose3d(in_size, out_size, kernel_size=3, stride=stride, padding=p, output_padding=op, bias=b), nn.ReLU(),
        nn.BatchNorm3d(out_size, eps=1e-2)
    )

  def forward(self, x, cat_in):
    return self.ct(torch.cat([x, cat_in], dim=1))

# ---------------------------------------------------------------------------------
class IRNv4_3DUNet(nn.Module):
  def __init__(self, in_size, depth=42, Na=1, Nb=1, Nc=1, bias=False, drop_p=0.0):
    """
    Na, Nb, Nc: number of times to run through layers A, B, C
    Inception-ResNet-v1/2 use 5, 10, 5
    """
    super(IRNv4_3DUNet, self).__init__()
    base = 32  # what does this look like in IRNv4?
    self.depth = depth
    self.Na, self.Nb, self.Nc = Na, Nb, Nc
    # down layers
    self.s1 = S1(in_size, base, b=bias)  # -> (B, 64, D, H, W)
    self.s2 = S2(64, base, b=bias)  # -> (B, 160, D, H, W)
    self.s3 = S3(160, base, b=bias)  # -> (B, 192, D, H, W)
    self.s4 = S4(192, b=bias)  # -> (B, 384, ...)
    self.a_n = [A(384, base, b=bias) for _ in range(Na)]  # (B, 384, ...)
    self.ra = redA(384, b=bias, drop_p=drop_p)  # -> (B, 1152, ...) this layer TRIPLES C
    self.b_n = [B(1152, b=bias) for _ in range(Nb)]  # -> (B, 1152, ...)
    self.rb = redB(1152, b=bias, drop_p=drop_p)  # (B, 2304, ...)  DOUBLES C
    self.c_n = [C(2304, b=bias, drop_p=drop_p) for _ in range(Nc)]  # -> (B, 2304, ...)

    # up layers -- possible to 'reverse' the inception blocks instead?
    # could replace the ConvTranspose3d with Upsample where MaxPool was used?
    # what about keeping the depth down after the down section? how does unet do this?
    self.rev_rb = revRL(2304, 1152, op=(0, 1, 1), b=bias, drop_p=drop_p)  # (B, 1152, ...
    self.rev_b = revLB(2304, 1152, b=bias, drop_p=drop_p)  # (B, 1152, ... CAT
    self.rev_ra = revRL(1152, 384, op=0, b=bias, drop_p=drop_p)  # (B, 384, ...
    self.rev_a = revLA(768, 384, b=bias, drop_p=drop_p)  # (B, 384, ... CAT
    self.rev_s4 = revS(384, 192, stride=2, b=bias)  # (B, 192, ...)
    self.rev_s3 = revScat(384, 160, stride=1, b=bias)  # (B, 160, ...)  CAT
    self.rev_s2 = revS(160, 64, stride=(1, 2, 2), p=1, op=(0, 1, 1), b=bias)  # (B, 128, ...)
    self.rev_s1 = revScat(128, in_size, stride=(1, 2, 2), p=1, op=(0, 1, 1), b=bias)  # (B, 5, ...)  CAT

    # collapse channels AND depth -> some form of FCL instead..? keep bias here?
    # attn UNet 'rebuilds' all depth to segment; do same and then FCL to 'smash' into surface precip?
    self.fin_linear = nn.Linear(self.depth, 1)  # uses only 42 + 1 params.. seems bad
    self.fin_conv = PartialConv3d(in_size, 1, kernel_size=1, stride=1, padding=0) # (B, 1, ...)
    self.final = PartialConv3d(in_size, 1, kernel_size=(self.depth, 1, 1), stride=1, padding=0)

    self.net_down = nn.ModuleList([self.s1, self.s2, self.s3, self.s4] + self.a_n + [self.ra] + self.b_n + [self.rb] + self.c_n)

    # do the same down here.?
    self.net_up = nn.ModuleList([
        self.rev_rb, self.rev_b, self.rev_ra, self.rev_a,
        self.rev_s4, self.rev_s3, self.rev_s2, self.rev_s1
    ])

  def forward(self, x):
    outs = []
    for i, step in enumerate(self.net_down):
      x = step(x)
      if i in [0, 2, 3 + self.Na, 3 + self.Na + 1 + self.Nb]:
        outs.append(x)
        print(x.shape)

    ctr = 0
    for j, step in enumerate(self.net_up):
      if (j + 1) % 2 == 0:
        ctr += 1
        print(x.shape, outs[-ctr].shape)
        x = step(x, outs[-ctr])
      else:
        x = step(x)

    # use FCL to collapse depth dim  ?
    # B, C, D, H, W = x.shape
    # x = x.permute(0, 1, 3, 4, 2).reshape(-1, self.depth)
    # x = self.fin_linear(x)
    # x = self.fin_conv(x.reshape(B, C, 1, H, W))
    
    # or just one big 3d conv??
    x = self.final(x)

    return x

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

