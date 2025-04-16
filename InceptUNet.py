import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

def main():
    net = IRNv4UNet(48)
    summary(net, input_size=(64, 48, 80, 144))

# ---------------------------------------------------------------------------------
class S1(nn.Module):
  def __init__(self, in_size, base=64, bias=False, drop_p=0.0):
    super(S1, self).__init__()
    # have this layer preserve size instead?

    self.stem_a = nn.Sequential(
        nn.Conv2d(in_size, base, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.Conv2d(base, base, kernel_size=3, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        #nn.Conv2d(in_size, base, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(),
        #nn.Conv2d(base, base, kernel_size=3, stride=1, padding=0, bias=False), nn.ReLU(),
        #nn.Conv2d(base, 2 * base, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),
        nn.BatchNorm2d(2 * base, eps=1e-2)
    )

  def forward(self, x):
    return self.stem_a(x)

# ---------------------------------------------------------------------------------
class S2(nn.Module):
  def __init__(self, in_size, base=64, bias=False, drop_p=0.0):
    super(S2, self).__init__()
    self.stem_b1 = nn.Sequential(
        nn.Conv2d(2 * base, 2 * base, kernel_size=3, stride=2, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(2 * base, eps=1e-2)
    )
    self.stem_b2 = nn.Sequential(
        nn.Conv2d(2 * base, 3 * base, kernel_size=3, stride=2, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(3 * base, eps=1e-2)
    )

  def forward(self, x):
    return torch.cat([self.stem_b1(x), self.stem_b2(x)], dim=1)

# ---------------------------------------------------------------------------------
class S3(nn.Module):
  def __init__(self, in_size, base=64, bias=False, drop_p=0.0):
    super(S3, self).__init__()
    self.stem_c1 = nn.Sequential(
        nn.Conv2d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(2 * base, 3 * base, kernel_size=3, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(3 * base, eps=1e-2)
    )
    self.stem_c2 = nn.Sequential(
        nn.Conv2d(in_size, 2 * base, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(2 * base, 2 * base, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=bias), nn.ReLU(),
        nn.Conv2d(2 * base, 2 * base, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=bias), nn.ReLU(),
        nn.Conv2d(2 * base, 3 * base, kernel_size=3, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(3 * base, eps=1e-2)
    )

  def forward(self, x):
    return torch.cat([self.stem_c1(x), self.stem_c2(x)], dim=1)

# ---------------------------------------------------------------------------------
class S4(nn.Module):
  def __init__(self, in_size, bias=False, drop_p=0.0):
    super(S4, self).__init__()
    self.stem_d1 = nn.Sequential(  # this last Conv is pad 0 if using odd input shape
        nn.Conv2d(in_size, in_size, kernel_size=3, stride=2, padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size, eps=1e-2)
    )
    self.stem_d2 = nn.MaxPool2d(2, stride=2, padding=0)

  def forward(self, x):
    return F.relu(torch.cat([self.stem_d1(x), self.stem_d2(x)], dim=1))

# ---------------------------------------------------------------------------------
class A(nn.Module):
  def __init__(self, in_size, base=64, bias=False, drop_p=0.0, lin_act=0.1):
    super(A, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv2d(in_size, base, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(base, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, base, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(base, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.Conv2d(in_size, base, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(base, 3 * base // 2, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.Conv2d(3 * base // 2, 2 * base, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(2 * base, eps=1e-2)
    )
    self.comb = nn.Conv2d(4 * base, in_size, kernel_size=1, stride=1, padding=0, bias=bias)
    self.final = nn.ReLU()

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)  # filter cat right?
    x_conv = self.lin_act * self.comb(x_conv) # linear activation scaling for stability

    return self.final(x + x_conv)  # residual connection

# ---------------------------------------------------------------------------------
class redA(nn.Module):
  def __init__(self, in_size, bias=False, drop_p=0.0):
    super(redA, self).__init__()
    self.b1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, in_size, kernel_size=3, stride=2, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.Conv2d(in_size, 3 * in_size // 2, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(3 * in_size // 2, 3 * in_size // 2, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.Conv2d(3 * in_size // 2, in_size, kernel_size=3, stride=2, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size, eps=1e-2)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))

# ---------------------------------------------------------------------------------
class B(nn.Module):
  def __init__(self, in_size, bias=False, drop_p=0.0, lin_act=0.1):
    super(B, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv2d(in_size, in_size // 8, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 8, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, in_size // 10, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(in_size // 10, in_size // 9, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=bias), nn.ReLU(),
        nn.Conv2d(in_size // 9, in_size // 8, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 8, eps=1e-2)
    )
    self.comb = nn.Conv2d(in_size // 4, in_size, kernel_size=1, stride=1, padding=0, bias=bias)
    self.final = nn.ReLU()

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class redB(nn.Module):
  def __init__(self, in_size, bias=False, drop_p=0.0):
    super(redB, self).__init__()
    self.b1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, 2 * in_size // 12, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(2 * in_size // 12, 5 * in_size // 12, kernel_size=3, stride=2, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(5 * in_size // 12, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.Conv2d(in_size, 2 * in_size // 12, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(2 * in_size // 12, 3 * in_size // 12, kernel_size=3, stride=2, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(3 * in_size // 12, eps=1e-2)
    )
    self.b4 = nn.Sequential(
        nn.Conv2d(in_size, 2 * in_size // 12, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(2 * in_size // 12, 3 * in_size // 12, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.Conv2d(3 * in_size // 12, 4 * in_size // 12, kernel_size=3, stride=2, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(4 * in_size // 12, eps=1e-2)
    )
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))

# ---------------------------------------------------------------------------------
class C(nn.Module):
  def __init__(self, in_size, bias=False, drop_p=0.0, lin_act=0.1):
    super(C, self).__init__()
    self.lin_act = lin_act
    self.b1 = nn.Sequential(
        nn.Conv2d(in_size, in_size // 10, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 10, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.Conv2d(in_size, in_size // 10, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.Conv2d(in_size // 10, in_size // 9, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=bias), nn.ReLU(),
        nn.Conv2d(in_size // 9, in_size // 8, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 8, eps=1e-2)
    )
    cat_size = in_size // 8 + in_size // 10
    self.comb = nn.Conv2d(cat_size, in_size, kernel_size=1, stride=1, padding=0, bias=bias)
    self.final = nn.Sequential(nn.ReLU(), nn.Dropout2d(p=drop_p))

  def forward(self, x):
    x_conv = torch.cat([self.b1(x), self.b2(x)], dim=1)
    x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
    return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class revLB(nn.Module):
  def __init__(self, in_size, out_size, bias=False, drop_p=0.0):
    super(revLB, self).__init__()
    self.b1 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 12, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.ConvTranspose2d(in_size // 12, in_size // 6, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 6, eps=1e-2)
    )
    self.b3 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.ConvTranspose2d(in_size //12, in_size // 8, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.ConvTranspose2d(in_size // 8, in_size // 4, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 4, eps=1e-2)
    )
    # self.ct = nn.Sequential(
    #     nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=1, padding=1), nn.ReLU(),
    #     nn.BatchNorm2d(out_size)
    # )
    self.final = nn.Dropout2d(p=drop_p)

  def forward(self, x, cat_in):
    comb = torch.cat([x, cat_in], dim=1)
    return self.final(torch.cat([self.b1(comb), self.b2(comb), self.b3(comb)], dim=1))

# ---------------------------------------------------------------------------------
class revLA(nn.Module):
  def __init__(self, in_size, out_size, bias=False, drop_p=0.0):
    super(revLA, self).__init__()
    self.b1 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 6, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 6, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=1, stride=1, padding=0, bias=bias), nn.ReLU(),
        nn.ConvTranspose2d(in_size // 12, in_size // 8, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.ConvTranspose2d(in_size // 8, in_size // 3, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // 3, eps=1e-2)
    )
    # self.ct = nn.Sequential(
    #     nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=1, padding=1), nn.ReLU(),
    #     nn.BatchNorm2d(out_size)
    # )
    self.final = nn.Dropout2d(p=drop_p)

  def forward(self, x, cat_in):
    comb = torch.cat([x, cat_in], dim=1)
    return self.final(torch.cat([self.b1(comb), self.b2(comb)], dim=1))

# ---------------------------------------------------------------------------------
class revRL(nn.Module):
  def __init__(self, in_size, out_size, bias=False, drop_p=0.0):
    super(revRL, self).__init__()
    # TODO: worth breaking this up like A/B reduction ones? 
    ratio = in_size // out_size
    hid = 2 * ratio
    self.b1 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=3, stride=2, padding=0, output_padding=1, bias=bias), nn.ReLU(),
        nn.ConvTranspose2d(in_size // 12, in_size // hid, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // hid, eps=1e-2)
    )
    self.b2 = nn.Sequential(
        nn.ConvTranspose2d(in_size, in_size // 12, kernel_size=3, stride=1, padding=1, bias=bias), nn.ReLU(),
        nn.ConvTranspose2d(in_size // 12, in_size // hid, kernel_size=3, stride=2, padding=0, output_padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(in_size // hid, eps=1e-2)
    ) 
    # self.ct = nn.Sequential(
    #     nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=0, output_padding=1), nn.ReLU(),
    #     nn.BatchNorm2d(out_size)
    # )
    self.final = nn.Dropout2d(p=drop_p)

  def forward(self, x):
    return self.final(torch.cat([self.b1(x), self.b2(x)], dim=1))

# ---------------------------------------------------------------------------------
class revS(nn.Module):
  def __init__(self, in_size, out_size, stride=1, padding=1, bias=False, drop_p=0.0):
    super(revS, self).__init__()
    self.ct = nn.Sequential(
        nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=stride, padding=padding, output_padding=1, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(out_size, eps=1e-2)
    )

  def forward(self, x):
    return self.ct(x)

# ---------------------------------------------------------------------------------
class revScat(nn.Module):
  def __init__(self, in_size, out_size, stride=1, bias=False, drop_p=0.0):
    super(revScat, self).__init__()
    self.ct = nn.Sequential(
        nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=stride, padding=0, bias=bias), nn.ReLU(),
        nn.BatchNorm2d(out_size, eps=1e-2)
    )

  def forward(self, x, cat_in):
    return self.ct(torch.cat([x, cat_in], dim=1))

# ---------------------------------------------------------------------------------
class IRNv4UNet(nn.Module):
  def __init__(self, in_size, Na=1, Nb=1, Nc=1, bias=False, drop_p=0.0, lin_act=0.1):
    """
    Na, Nb, Nc: number of times to run through layers A, B, C
    Inception-ResNet-v1/2 use 5, 10, 5
    """
    super(IRNv4UNet, self).__init__()
    base = 64 # in_size
    self.Na, self.Nb, self.Nc = Na, Nb, Nc
    # down layers
    self.s1 = S1(in_size, base, bias=bias, drop_p=drop_p)  # (B, 128, 45, 45) OR (128, 94, 94)
    self.s2 = S2(2 * base, base, bias=bias, drop_p=drop_p)  # (B, 640, 22, 22) OR (320, 46, 46)
    self.s3 = S3(5 * base, base, bias=bias, drop_p=drop_p)  # (B, 768, 20, 20) (768, 44, 44) OR (384, 44, 44)
    self.s4 = S4(6 * base, bias=bias, drop_p=drop_p)  # (B, 1536, 10, 10) (1536, 22, 22) OR (768, 22, 22)
    self.a_n = [A(12 * base, base, bias=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Na)]  # (B, 1536, 10, 10) (1536, 22, 22) OR (768, 22, 22)
    self.ra = redA(12 * base, bias=bias, drop_p=drop_p)  # (B, 4608, 4, 4) (4608, 10, 10) OR (2304, 10, 10)
    self.b_n = [B(36 * base, bias=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Nb)] # (4608, 10, 10) OR (2304, 10, 10)
    self.rb = redB(36 * base, bias=bias, drop_p=drop_p) # (9216, 4, 4)  consider removing this? OR (4608, 4, 4)
    self.c_n = [C(72 * base, bias=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Nc)] # (4608, 4, 4)

    # up layers -- possible to 'reverse' the inception blocks instead?
    # could replace the ConvTranspose2d with Upsample where small change in input size?
    self.rev_rb = revRL(72 * base, 36 * base, bias=bias, drop_p=drop_p)  # (B, 2304, 10, 10)
    self.rev_b = revLB(72 * base, 36 * base, bias=bias, drop_p=drop_p)  # (B, 2304, 10, 10)
    self.rev_ra = revRL(36 * base, 12 * base, bias=bias, drop_p=drop_p)  # (B, 768, 22, 22)
    self.rev_a = revLA(24 * base, 12 * base, bias=bias, drop_p=drop_p)  # (B, 768, 22, 22)
    self.rev_s4 = revS(12 * base, 6 * base, stride=2, padding=1, bias=bias, drop_p=drop_p)  # (B, 384, 44, 44)
    self.rev_s3 = revScat(12 * base, 5 * base, stride=1, bias=bias, drop_p=drop_p)  # (B, 320, 46, 46)
    self.rev_s2 = revS(5 * base, 2 * base, stride=2, padding=0, bias=bias, drop_p=drop_p)  # (B, 128, 94, 94)
    self.rev_s1 = revScat(4 * base, in_size, stride=1, bias=bias, drop_p=drop_p)  # (B, 64, 96, 96)

    # TODO: this was previously 3x3 kernel
    self.final = nn.Conv2d(in_size, 1, kernel_size=1, stride=1, padding=0, bias=bias) # (B, 1, 96, 96)

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
      #print(step, x.shape)

    ctr = 0
    for j, step in enumerate(self.net_up):
      if (j + 1) % 2 == 0:
        ctr += 1
        x = step(x, outs[-ctr])
        #print(step, x.shape, outs[-ctr].shape)
      else:
        x = step(x)
        #print(step, x.shape)

    return self.final(x)

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

