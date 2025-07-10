import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from Components import *

PRINT = False
# ---------------------------------------------------------------------------------
def main():
    net = UNet(depth=16).cuda()
    test = torch.ones(1, 5, 16, 80, 144).cuda()
    net(test)
    #print(net)  doing 54, 42 is just too small with all the downsampling..
    s = (64, 5, 16, 80, 144)
    summary(net, input_size=s)
    # TODO: handle padding/output padding stuff better! this will only work for known input shape
    #  need to somehow figure out even vs. odd shapes how to handle padding

# ---------------------------------------------------------------------------------
def select_pad(shape_in, shape_out):
    D_in, H_in, W_in = shape_in[2:]
    D_out, H_out, W_out = shape_out[2:] 

def calc_shape_out(shape_in, p, d, k, s):
    return ((shape_in + 2 * p - d * (k - 1) - 1) / s + 1).__floor__()

def printit(*args):
    print(*args) if PRINT else None

# ---------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, c):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            Conv3(c, k=3, s=1, p=1),
            Conv3(c, k=3, s=1, p=1)
        )

    def forward(self, x, cat_in=None):
        if cat_in is not None:
            x = self.net(torch.cat([x, cat_in], dim=1))
        else:
            x = self.net(x)
        printit(f'conv {x.shape}')
        return x

class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.net = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.net(x)
        printit(f'down {x.shape}')
        return x

class Upsample(nn.Module):
    def __init__(self, c):
        super(Upsample, self).__init__()
        #self.net = TConv3(c, k=2, s=2, p=0)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv3(c, k=3, s=1, p=1)
        )

    def forward(self, x):
        x = self.net(x)
        printit(f'up {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, depth=16):
        super(UNet, self).__init__()
        self.depth = depth
        init_c = 32
        # down layers
        self.down = nn.ModuleList([
            ConvBlock(init_c), Downsample(), ConvBlock(2 * init_c), Downsample(), ConvBlock(4 * init_c), Downsample(),
            ConvBlock(8 * init_c), Downsample(), ConvBlock(16 * init_c)
        ])
        # up layers
        self.up = nn.ModuleList([
            Upsample(8 * init_c), ConvBlock(8 * init_c), Upsample(4 * init_c), ConvBlock(4 * init_c),
            Upsample(2 * init_c), ConvBlock(2 * init_c), Upsample(init_c), ConvBlock(init_c)
        ])
        self.red_depth = Conv3(init_c, k=(depth, 1, 1), s=1, p=0)
        self.final = Conv2(1, k=1, s=1, p='same')

    def forward(self, x):
        outs = []
        for i, step in enumerate(self.down):
            x = step(x)
            if i % 2 == 0:
                outs.append(x)

        ctr = 0
        for j, step in enumerate(self.up):
            if j + 1 % 2 == 0:
                ctr += 1
                x = step(x, outs[-ctr])
            else:
                x = step(x)

        x = self.red_depth(x).squeeze(2)
        printit(f'red depth {x.shape}')
        x = self.final(x)
        printit(f'final {x.shape}')
        return x

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

