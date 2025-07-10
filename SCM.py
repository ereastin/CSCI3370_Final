import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from Components import *

PRINT = False
ROI = False
# all Dx1x1 conv except reductions?
# ---------------------------------------------------------------------------------
def main():
    bias = False

    net = SCM(5, depth=16, bias=bias).cuda()
    #print(net)
    test = torch.ones(1, 5, 16, 80, 144).cuda()
    net(test)
    #print(net)  doing 54, 42 is just too small with all the downsampling..
    s = (64, 5, 16, 80, 144)
    summary(net, input_size=s)
    # TODO: handle padding/output padding stuff better! this will only work for known input shape
    #  need to somehow figure out even vs. odd shapes how to handle padding

# ---------------------------------------------------------------------------------
# unused, are these valuable for realtime adjustment or not really
def select_pad(shape_in, shape_out):
    D_in, H_in, W_in = shape_in[2:]
    D_out, H_out, W_out = shape_out[2:] 

def calc_shape_out(shape_in, p, d, k, s):
    return ((shape_in + 2 * p - d * (k - 1) - 1) / s + 1).__floor__()

def printit(*args):
    print(*args) if PRINT else None

class ResBlock(nn.Module):
    def __init__(self, c, b=False):
        super(ResBlock, self).__init__()
        self.net =  nn.Sequential(
            Conv3(c, k=(3, 1, 1), s=1, p='same', b=b),
            Conv3(c, k=(5, 1, 1), s=1, p='same', b=b),
            Conv3(c, k=(7, 1, 1), s=1, p='same', b=b),
        )

    def forward(self, x):
        return x + self.net(x)

class Reduce(nn.Module):
    def __init__(self, c, b=False):
        self.net = Conv3(c, k=3, s=2, p=0, b=b)

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------------
class SCM(nn.Module):
    def __init__(self, in_size, depth=16, bias=False, drop_p=0.0, lin_act=0.1):
        super(SCM, self).__init__()
        self.start = nn.LazyConv3d(4 * in_size, kernel_size=1, stride=1, padding=0, bias=bias)
        self.vgg = nn.ModuleList([ResBlock(4 * in_size) for _ in range(16)])
        # probs just use a linear layer instead.?
        self.twod = nn.LazyConv3d(in_size, kernel_size=(depth, 1, 1), stride=1, padding=0, bias=bias)
        self.final = nn.LazyConv2d(1, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.start(x)
        for step in self.vgg:
            x = step(x)
        print(x.shape)
        x = self.twod(x).squeeze(2)
        print(x.shape)
        x = self.final(x)
        return x

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

