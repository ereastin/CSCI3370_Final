import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# ---------------------------------------------------------------------------------
def main():
    na = 3
    net = IRNv4_3D(6, depth=35, Na=na, Nb=2*na, Nc=na, base=24, bias=False, drop_p=0.0, lin_act=0.1)
    s = (64, 6, 35, 80, 144)
    summary(net, input_size=s)
    # TODO: handle padding/output padding stuff better! this will only work for known input shape
    #  need to somehow figure out even vs. odd shapes how to handle padding

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
    # dilation??
    ka = 11
    kb = 7
    kc = 3
    self.stem_a = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=ka, stride=1, padding=5, bias=b), nn.ReLU(),
        nn.Conv3d(base, base, kernel_size=(3, 13, 3), stride=(2, 1, 2), padding=1, bias=b), nn.ReLU(),
        #nn.Conv3d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=1e-2), nn.Dropout3d(p=drop_p)
    )
    self.stem_b = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=kb, stride=1, padding=3, bias=b), nn.ReLU(),
        nn.Conv3d(base, base, kernel_size=(3, 13, 3), stride=(2, 1, 2), padding=1, bias=b), nn.ReLU(),
        #nn.Conv3d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=1e-2), nn.Dropout3d(p=drop_p)
    )
    self.stem_c = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=kc, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.Conv3d(base, base, kernel_size=(3, 13, 3), stride=(2, 1, 2), padding=1, bias=b), nn.ReLU(),
        #nn.Conv3d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=1e-2), nn.Dropout3d(p=drop_p)
    )

  def forward(self, x):
    return torch.cat([self.stem_a(x), self.stem_b(x), self.stem_c(x)], dim=1)

# ---------------------------------------------------------------------------------
class S2(nn.Module):
  def __init__(self, in_size, base=32, b=False):
    super(S2, self).__init__()
    self.stem_b1 = nn.Sequential(
        nn.MaxPool3d(kernel_size=(3, 9, 3), stride=(1, 1, 1), padding=1)
    )
    self.stem_b2 = nn.Sequential(
        nn.Conv3d(in_size, 3 * base, kernel_size=(3, 9, 3), stride=(1, 1, 1), padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=1e-2)
    )

  def forward(self, x):
      return torch.cat([self.stem_b1(x), self.stem_b2(x)], dim=1)

# ---------------------------------------------------------------------------------
class S3(nn.Module):
  def __init__(self, in_size, base=32, b=False):
    super(S3, self).__init__()
    self.stem_c1 = nn.Sequential(
        nn.Conv3d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv3d(2 * base, 3 * base, kernel_size=(3, 7, 3), stride=1, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=1e-2)
    )
    self.stem_c2 = nn.Sequential(
        nn.Conv3d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv3d(2 * base, 2 * base, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0), bias=b), nn.ReLU(),
        nn.Conv3d(2 * base, 2 * base, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3), bias=b), nn.ReLU(),
        nn.Conv3d(2 * base, 3 * base, kernel_size=(3, 7, 3), stride=1, padding=(1, 0, 0), bias=b), nn.ReLU(),
        nn.BatchNorm3d(3 * base, eps=1e-2)
    )

  def forward(self, x):
    return torch.cat([self.stem_c1(x), self.stem_c2(x)], dim=1)

# ---------------------------------------------------------------------------------
class S4(nn.Module):
  def __init__(self, in_size, base=32, b=False):
    super(S4, self).__init__()
    self.stem_d1 = nn.Sequential(
        nn.Conv3d(in_size, in_size, kernel_size=(3, 5, 3), stride=(2, 1, 2), padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size, eps=1e-2)
    )
    self.stem_d2 = nn.MaxPool3d(kernel_size=(3, 5, 3), stride=(2, 1, 2), padding=0)

  def forward(self, x):
    d1 = self.stem_d1(x)
    d2 = self.stem_d2(x)
    #print(d1.shape, d2.shape)
    return F.relu(torch.cat([d1, d2], dim=1))

# ---------------------------------------------------------------------------------
class A(nn.Module):
  def __init__(self, in_size, base=32, b=False, lin_act=0.1):
    super(A, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv3d(base, base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(base, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.Conv3d(in_size, base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv3d(base, 3 * base // 2, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.Conv3d(3 * base // 2, 2 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.BatchNorm3d(2 * base, eps=1e-2)
    )
    self.comb = nn.Conv3d(4 * base, in_size, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.ReLU()

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)  # filter cat right?
    x_conv = self.lin_act * self.comb(x_conv) # linear activation scaling for stability

    return self.final(x + x_conv)  # residual connection

# ---------------------------------------------------------------------------------
class redA(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(redA, self).__init__()
    self.b1 = nn.MaxPool3d(kernel_size=7, stride=(2, 1, 2), padding=(0, 0, 2))
    self.b2 = nn.Sequential(
        nn.Conv3d(in_size, in_size // 2, kernel_size=7, stride=(2, 1, 2), padding=(0, 0, 2), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 2, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.Conv3d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv3d(8 * base, 8 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.Conv3d(8 * base, in_size // 2, kernel_size=7, stride=(2, 1, 2), padding=(0, 0, 2), bias=b), nn.ReLU(),
        nn.BatchNorm3d(in_size // 2, eps=1e-2)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))

  def forward(self, x):
    b1 = self.b1(x)
    b2 = self.b2(x)
    b3 = self.b3(x)
    #print(b1.shape, b2.shape, b3.shape)
    cat = torch.cat([b1, b2, b3], dim=1)
    s = cat.shape
    #print(s)
    cat = cat.reshape(s[0], -1, s[3], s[4])
    #print(cat.shape)
    return self.final(cat)

# ---------------------------------------------------------------------------------
class B(nn.Module):
  def __init__(self, in_size, base=32, b=False, lin_act=0.1):
    super(B, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv2d(in_size, 6 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm2d(6 * base, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, 4 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv2d(4 * base, 5 * base, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=b), nn.ReLU(),
        nn.Conv2d(5 * base, 6 * base, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=b), nn.ReLU(),
        nn.BatchNorm2d(6 * base, eps=1e-2)
    )
    self.comb = nn.Conv2d(12 * base, in_size, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.ReLU()

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class redB(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0):
    super(redB, self).__init__()
    self.b1 = nn.MaxPool2d(kernel_size=(5, 3), stride=(1, 1), padding=(0, 1))
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv2d(8 * base, 9 * base, kernel_size=(5, 3), stride=(1, 1), padding=(0, 1), bias=b), nn.ReLU(),
        nn.BatchNorm2d(9 * base, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.Conv2d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv2d(8 * base, 9 * base, kernel_size=(5, 3), stride=(1, 1), padding=(0, 1), bias=b), nn.ReLU(),
        nn.BatchNorm2d(9 * base, eps=1e-2)
    )
    self.b4 = nn.Sequential(
        nn.Conv2d(in_size, 8 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv2d(8 * base, 9 * base, kernel_size=3, stride=1, padding=1, bias=b), nn.ReLU(),
        nn.Conv2d(9 * base, 10 * base, kernel_size=(5, 3), stride=(1, 1), padding=(0, 1), bias=b), nn.ReLU(),
        nn.BatchNorm2d(10 * base, eps=1e-2)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))

  def forward(self, x):
    b1 = self.b1(x)
    b2 = self.b2(x)
    b3 = self.b3(x)
    b4 = self.b4(x)
    #print(b1.shape, b2.shape, b3.shape, b4.shape)
    cat = torch.cat([b1, b2, b3, b4], dim=1)
    return self.final(cat)

# ---------------------------------------------------------------------------------
class C(nn.Module):
  def __init__(self, in_size, base=32, b=False, drop_p=0.0, lin_act=0.1):
    super(C, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv2d(in_size, 6 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.BatchNorm2d(6 * base, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, 6 * base, kernel_size=1, stride=1, padding=0, bias=b), nn.ReLU(),
        nn.Conv2d(6 * base, 7 * base, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=b), nn.ReLU(),
        nn.Conv2d(7 * base, 8 * base, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=b), nn.ReLU(),
        nn.BatchNorm2d(8 * base, eps=1e-2)
    )
    self.comb = nn.Conv2d(14 * base, in_size, kernel_size=1, stride=1, padding=0, bias=b)
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout3d(p=drop_p))

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class IRNv4_3D(nn.Module):
  def __init__(self, in_size, depth=42, Na=1, Nb=1, Nc=1, base=32, bias=False, drop_p=0.0, lin_act=0.1):
    """
    Na, Nb, Nc: number of times to run through layers A, B, C
    Inception-ResNet-v1/2 use 5, 10, 5
    """
    super(IRNv4_3D, self).__init__()
    self.depth = depth
    self.Na, self.Nb, self.Nc = Na, Nb, Nc
    # down layers
    self.s1 = S1(in_size, base, b=bias)  # -> (B, 64, , H, W)
    self.s2 = S2(3 * base, base, b=bias)  # -> (B, 160, , H, W)
    self.s3 = S3(6 * base, base, b=bias)  # -> (B, 192, , H, W)
    self.s4 = S4(6 * base, base, b=bias)  # -> (B, 384, D, H, W)
    self.a_n = [A(12 * base, base, b=bias, lin_act=lin_act) for _ in range(Na)]  # (B, 384, D, H, W)
    self.ra = redA(12 * base, base, b=bias, drop_p=drop_p)  # -> (B, 1152, D, H, W) this layer TRIPLES C
    self.b_n = [B(24 * base, base, b=bias, lin_act=lin_act) for _ in range(Nb)]  # -> (B, 1152, D, H, W)
    self.rb = redB(24 * base, base, b=bias, drop_p=drop_p)  # (B, 2048, 1, H, W)
    self.c_n = [C(52 * base, base, b=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Nc)]  # -> (B, 2048, 1, H, W)

    self.final = nn.Sequential(
        nn.Conv2d(52 * base, 26 * base, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(26 * base, 13 * base, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(13 * base, 6 * base, kernel_size=2, stride=1, padding=0),
        nn.Conv2d(6 * base, 1, kernel_size=1, stride=1, padding=0),
    )

    self.net_down = nn.ModuleList([self.s1, self.s2, self.s3, self.s4] + self.a_n + [self.ra] + self.b_n + [self.rb] + self.c_n)

  def forward(self, x):
    #outs = []
    for i, step in enumerate(self.net_down):
      x = step(x)
      #if i in [0, 2, 3 + self.Na, 3 + self.Na + 1 + self.Nb]:
      #  outs.append(x)
    
    #print(step, x.shape)
    # TODO how do we squis to 2d?
    x = self.final(x)
    return x

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

