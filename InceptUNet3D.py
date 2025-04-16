import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

EPS = 1e-5  # this is the default for BatchNorm
# ---------------------------------------------------------------------------------
def main():
    na = 1
    net = IRNv4_3DUNet(6, depth=35, Na=na, Nb=2*na, Nc=na, base=32, bias=False, drop_p=0.0, lin_act=0.1)
    s = (256, 6, 35, 80, 144)
    summary(net, input_size=s)
    # TODO: handle padding/output padding stuff better! this will only work for known input shape
    #  need to somehow figure out even vs. odd shapes how to handle padding
    # TODO: add linear activation scaling to revRA, revRB at residual addition
    # also add dropout everywhere?
    # also check the batchnorm eps bs? what even is that?

# unused, are these valuable for realtime adjustment or not really
def select_pad(shape_in, shape_out):
    D_in, H_in, W_in = shape_in[2:]
    D_out, H_out, W_out = shape_out[2:] 

def calc_shape_out(shape_in, p, d, k, s):
    return ((shape_in + 2 * p - d * (k - 1) - 1) / s + 1).__floor__()

# ---------------------------------------------------------------------------------
class S1(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(S1, self).__init__()

    self.stem_a = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=3, stride=(2, 2, 2), padding=1, bias=b), nn.ReLU(), nn.BatchNorm3d(base, eps=EPS),
        nn.Conv3d(base, base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(), nn.BatchNorm3d(base, eps=EPS),
        nn.Conv3d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(2 * base, eps=EPS), nn.Dropout3d(p=drop_p)
    )

  def forward(self, x):
    return self.stem_a(x)

# ---------------------------------------------------------------------------------
class S2(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(S2, self).__init__()
    self.stem_b1 = nn.Sequential(
        nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
    )
    self.stem_b2 = nn.Sequential(
        nn.Conv3d(2 * base, 3 * base, kernel_size=3, stride=(1, 2, 2), padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=EPS)
    )
    self.drop = nn.Dropout3d(p=drop_p)

  def forward(self, x):
    return self.drop(torch.cat([self.stem_b1(x), self.stem_b2(x)], dim=1))

# ---------------------------------------------------------------------------------
class S3(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(S3, self).__init__()
    self.stem_c1 = nn.Sequential(
        nn.Conv3d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(2 * base, eps=EPS),
        nn.Conv3d(2 * base, 3 * base, kernel_size=3, stride=1, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=EPS)
    )
    self.stem_c2 = nn.Sequential(
        nn.Conv3d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(2 * base, eps=EPS),
        # TODO: check these, thoughts on >1 k_size for depth dim?
        nn.Conv3d(2 * base, 2 * base, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0), bias=b), nn.ReLU(), nn.BatchNorm3d(2 * base, eps=EPS),
        nn.Conv3d(2 * base, 2 * base, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3), bias=b), nn.ReLU(), nn.BatchNorm3d(2 * base, eps=EPS),
        nn.Conv3d(2 * base, 3 * base, kernel_size=3, stride=1, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=EPS)
    )
    self.drop = nn.Dropout3d(p=drop_p)

  def forward(self, x):
    return self.drop(torch.cat([self.stem_c1(x), self.stem_c2(x)], dim=1))

# ---------------------------------------------------------------------------------
class S4(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(S4, self).__init__()
    self.stem_d1 = nn.Sequential(
        nn.Conv3d(in_size, in_size, kernel_size=3, stride=2, padding=(1, 1, 1), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size, eps=EPS)
    )
    self.stem_d2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    self.drop = nn.Dropout3d(p=drop_p)

  def forward(self, x):
    # TODO: should this relu be here..?
    # return F.relu(torch.cat([self.stem_d1(x), self.stem_d2(x)], dim=1))
    return self.drop(torch.cat([self.stem_d1(x), self.stem_d2(x)], dim=1))

# ---------------------------------------------------------------------------------
class A(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0, lin_act=0.1):
    super(A, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=EPS)
    )
    self.b2 = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(base, eps=EPS),
        nn.Conv3d(base, base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=EPS)
    )
    self.b3 = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(base, eps=EPS),
        nn.Conv3d(base, 3 * base // 2, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(), nn.BatchNorm3d(3 * base // 2, eps=EPS),
        nn.Conv3d(3 * base // 2, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(2 * base, eps=EPS)
    )
    self.comb = nn.Conv3d(4 * base, in_size, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)  # filter cat right?
    x_conv = self.lin_act * self.comb(x_conv) # linear activation scaling for stability

    return self.final(x + x_conv)  # residual connection

# ---------------------------------------------------------------------------------
class redA(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(redA, self).__init__()
    self.b1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=(1, 0, 0))
    self.b2 = nn.Sequential(
        nn.Conv3d(in_size, in_size, kernel_size=3, stride=2, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size, eps=EPS)
    )
    self.b3 = nn.Sequential(
        nn.Conv3d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(8 * base, eps=EPS),
        nn.Conv3d(8 * base, 8 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(), nn.BatchNorm3d(8 * base, eps=EPS),
        nn.Conv3d(8 * base, in_size, kernel_size=3, stride=2, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size, eps=EPS)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))

# ---------------------------------------------------------------------------------
class B(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0, lin_act=0.1):
    super(B, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv3d(in_size, 6 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(6 * base, eps=EPS)
    )
    self.b2 = nn.Sequential(
        nn.Conv3d(in_size, 4 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(4 * base, eps=EPS),
        nn.Conv3d(4 * base, 5 * base, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0), bias=b), nn.ReLU(), nn.BatchNorm3d(5 * base, eps=EPS),
        nn.Conv3d(5 * base, 6 * base, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3), bias=b), nn.ReLU(),
        nn.BatchNorm3d(6 * base, eps=EPS)
    )
    self.comb = nn.Conv3d(12 * base, 36 * base, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class redB(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(redB, self).__init__()
    self.b1 = nn.MaxPool3d(kernel_size=(5, 3, 3), stride=2, padding=(0, 0, 0))
    self.b2 = nn.Sequential(
        nn.Conv3d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(8 * base, eps=EPS),
        nn.Conv3d(8 * base, 9 * base, kernel_size=(5, 3, 3), stride=2, padding=(0, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(9 * base, eps=EPS)
    )
    self.b3 = nn.Sequential(
        nn.Conv3d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(8 * base, eps=EPS),
        nn.Conv3d(8 * base, 9 * base, kernel_size=(5, 3, 3), stride=2, padding=(0, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(9 * base, eps=EPS)
    )
    self.b4 = nn.Sequential(
        nn.Conv3d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(8 * base, eps=EPS),
        nn.Conv3d(8 * base, 9 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(), nn.BatchNorm3d(9 * base, eps=EPS),
        nn.Conv3d(9 * base, 10 * base, kernel_size=(5, 3, 3), stride=2, padding=(0, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(10 * base, eps=EPS)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))

# ---------------------------------------------------------------------------------
class C(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0, lin_act=0.1):
    super(C, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv3d(in_size, 6 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(6 * base, eps=EPS)
    )
    self.b2 = nn.Sequential(
        nn.Conv3d(in_size, 6 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm3d(6 * base, eps=EPS),
        nn.Conv3d(6 * base, 7 * base, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=b), nn.ReLU(), nn.BatchNorm3d(7 * base, eps=EPS),
        nn.Conv3d(7 * base, 8 * base, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(8 * base, eps=EPS)
    )
    self.comb = nn.Conv3d(14 * base, 64 * base, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class revLB(nn.Module):
  def __init__(self, in_size, out_size, b=False, drop_p=0.0):
    super(revLB, self).__init__()
    self.b1 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm2d(in_size // 12, eps=EPS)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm2d(in_size // 12, eps=EPS),
        nn.ConvTranspose2d(in_size // 12, in_size // 6, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm2d(in_size // 6, eps=EPS)
    )
    self.b3 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm2d(in_size // 12, eps=EPS),
        nn.ConvTranspose2d(in_size //12, in_size // 8, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(), nn.BatchNorm2d(in_size // 8, eps=EPS),
        nn.ConvTranspose2d(in_size // 8, in_size // 4, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm2d(in_size // 4, eps=EPS)
    )
    # TODO: include a 1x1 conv in these ones?
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))
    self.squish = nn.Conv3d(out_size, out_size, kernel_size=(5, 1, 1), stride=1, padding=0) 

  def forward(self, x, cat_in):
    cat_in = self.squish(cat_in).squeeze(2)
    # print(f'cat_in revLB: {cat_in.shape}')
    comb = torch.cat([x, cat_in], dim=1)
    return self.final(torch.cat([self.b1(comb), self.b2(comb), self.b3(comb)], dim=1))

# ---------------------------------------------------------------------------------
class revLA(nn.Module):
  def __init__(self, in_size, out_size, b=False, drop_p=0.0):
    super(revLA, self).__init__()
    self.b1 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 6, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm2d(in_size // 6, eps=EPS)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(), nn.BatchNorm2d(in_size // 12, eps=EPS),
        nn.ConvTranspose2d(in_size // 12, in_size // 8, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(), nn.BatchNorm2d(in_size // 8, eps=EPS),
        nn.ConvTranspose2d(in_size // 8, in_size // 3, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm2d(in_size // 3, eps=EPS)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))
    self.squish = nn.Conv3d(out_size, out_size, kernel_size=(9, 1, 1), stride=1, padding=0) 

  def forward(self, x, cat_in):
    cat_in = self.squish(cat_in).squeeze(2)
    # print(f'cat_in revLA: {cat_in.shape}')
    comb = torch.cat([x, cat_in], dim=1)
    return self.final(torch.cat([self.b1(comb), self.b2(comb)], dim=1))

# ---------------------------------------------------------------------------------
class revRL(nn.Module):
  def __init__(self, in_size, out_size, op=1, b=False, drop_p=0.0):
    super(revRL, self).__init__()
    # TODO: worth breaking this up like A/B reduction ones? 
    half_out = out_size // 2
    self.b1 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=3, stride=2, padding=0, output_padding=op, bias=b), nn.ReLU(), nn.BatchNorm2d(in_size // 12, eps=EPS),
        nn.ConvTranspose2d(in_size // 12, half_out, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm2d(half_out, eps=EPS)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(), nn.BatchNorm2d(in_size // 12, eps=EPS),
        nn.ConvTranspose2d(in_size // 12, half_out, kernel_size=3, stride=2, padding=0, output_padding=op, bias=b), nn.ReLU(),
        nn.BatchNorm2d(half_out, eps=EPS)
    ) 
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x)], dim=1))

# ---------------------------------------------------------------------------------
class revS(nn.Module):
  def __init__(self, in_size, out_size, stride=2, p=1, op=1, b=False, drop_p=0.0):
    super(revS, self).__init__()
    self.ct = nn.Sequential(
        nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=stride, padding=p, output_padding=op, bias=b), nn.ReLU(),
        nn.BatchNorm2d(out_size, eps=EPS)
    )
    self.drop = nn.Dropout2d(p=drop_p)

  def forward(self, x):
    return self.drop(self.ct(x))

# ---------------------------------------------------------------------------------
class revScat(nn.Module):
  def __init__(self, in_size, out_size, stride=1, p=(1, 0), dd=1, op=0, b=False, drop_p=0.0, ks=(1, 2, 2), kp=(0, 1, 1), d=1):
    super(revScat, self).__init__()
    self.ct = nn.Sequential(
        nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=stride, padding=p, dilation=dd, output_padding=op, bias=b), nn.ReLU(),
        nn.BatchNorm2d(out_size, eps=EPS)
    )
    self.squish = nn.Conv3d(in_size // 2, in_size // 2, kernel_size=(18, 3, 3), stride=ks, padding=kp, dilation=d)
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))

  def forward(self, x, cat_in):
    cat_in = self.squish(cat_in).squeeze(2)
    #print(f'scat shape: {cat_in.shape}')
    return self.final(self.ct(torch.cat([x, cat_in], dim=1)))

# ---------------------------------------------------------------------------------
class IRNv4_3DUNet(nn.Module):
  def __init__(self, in_size, depth=42, Na=1, Nb=1, Nc=1, base=32, bias=False, drop_p=0.0, lin_act=0.1):
    """
    Na, Nb, Nc: number of times to run through layers A, B, C
    Inception-ResNet-v1/2 use 5, 10, 5
    """
    super(IRNv4_3DUNet, self).__init__()
    self.depth = depth
    self.Na, self.Nb, self.Nc = Na, Nb, Nc
    # down layers
    self.s1 = S1(in_size, base, b=bias, drop_p=drop_p)  # -> (B, 64, , H, W)
    self.s2 = S2(2 * base, base, b=bias, drop_p=drop_p)  # -> (B, 160, , H, W)
    self.s3 = S3(5 * base, base, b=bias, drop_p=drop_p)  # -> (B, 192, , H, W)
    self.s4 = S4(6 * base, base, b=bias, drop_p=drop_p)  # -> (B, 384, D, H, W)
    self.a_n = [A(12 * base, base, b=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Na)]  # (B, 384, D, H, W)
    self.ra = redA(12 * base, base, b=bias, drop_p=drop_p)  # -> (B, 1152, D, H, W) this layer TRIPLES C
    self.b_n = [B(36 * base, base, b=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Nb)]  # -> (B, 1152, D, H, W)
    self.rb = redB(36 * base, base, b=bias, drop_p=drop_p)  # (B, 2048, 1, H, W)
    self.c_n = [C(64 * base, base, b=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Nc)]  # -> (B, 2048, 1, H, W)

    # up layers -- possible to 'reverse' the inception blocks instead?
    # could replace the ConvTranspose2d with Upsample where MaxPool was used?
    # ^^ would reduce the number of params here
    # also consider just removing the UNet cat parts
    self.rev_rb = revRL(64 * base, 36 * base, b=bias, drop_p=drop_p)  # (B, 1152, ...
    self.rev_b = revLB(72 * base, 36 * base, b=bias, drop_p=drop_p)  # (B, 1152, ... CAT
    self.rev_ra = revRL(36 * base, 12 * base, op=0, b=bias, drop_p=drop_p)  # (B, 384, ...
    self.rev_a = revLA(24 * base, 12 * base, b=bias, drop_p=drop_p)  # (B, 384, ... CAT
    self.rev_s4 = revS(12 * base, 6 * base, stride=(1, 1), p=1, op=0, b=bias, drop_p=drop_p)  # (B, 192, ...)
    self.rev_s3 = revScat(12 * base, 5 * base, stride=(2, 1), p=1, b=bias, drop_p=drop_p)  # (B, 160, ...)  CAT
    self.rev_s2 = revS(5 * base, 2 * base, stride=(2, 1), p=(0, 1), op=(1, 0), b=bias, drop_p=drop_p)  # (B, 128, ...)
    self.rev_s1 = revScat(4 * base, in_size, stride=(1, 1), p=(0, 1), op=0, b=bias, drop_p=drop_p, ks=(1, 1, 4), kp=0, d=(1, 2, 3))  # (B, 5, ...)  CAT

    self.ff = nn.ConvTranspose2d(in_size, in_size, kernel_size=(8, 3), stride=1, padding=(0, 1), bias=bias)    
    self.final = nn.Conv2d(in_size, 1, kernel_size=3, stride=1, padding=0, bias=bias)

    self.net_down = nn.ModuleList([self.s1, self.s2, self.s3, self.s4] + self.a_n + [self.ra] + self.b_n + [self.rb] + self.c_n)

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
      #print(step, x.shape)

    x = x.squeeze(2)
    ctr = 0
    for j, step in enumerate(self.net_up):
      if (j + 1) % 2 == 0:
        ctr += 1
        #print(step, x.shape, outs[-ctr].shape)
        x = step(x, outs[-ctr])
      else:
        x = step(x)
        #print(step, x.shape)
    #print('FINAL', x.shape)
    x = self.final(F.relu(self.ff(x)))
    return x

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

