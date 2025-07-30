import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from Components import *

PRINT = False
C, D, H, W = 6, 16, 54, 42
N_CLASS = C
# ---------------------------------------------------------------------------------
def main():
    net = MultiUNet(n_vars=C, depth=D, spatial_dim=2, init_c=32, embedding_dim=32).cuda()
    test = torch.ones(1, C, D, H, W).cuda()
    net(test)
    s = (64, C, D, H, W)
    summary(net, input_size=s)
    #snglnet = UNet(dim=2)
    #s = (64, D, H, W)
    #summary(snglnet, input_size=s)
    # TODO: handle padding/output padding stuff better! this will only work for known input shape
    #  need to somehow figure out even vs. odd shapes how to handle padding

def printit(*args):
    print(*args) if PRINT else None

# ---------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, c, dim=3, bias=False):
        super(ConvBlock, self).__init__()
        if dim == 3:
            self.net = nn.Sequential(
                Conv3(c, k=3, s=1, p=1, b=bias),
                Conv3(c, k=3, s=1, p=1, b=bias)
            )
        else:
            self.net = nn.Sequential(
                Conv2(c, k=3, s=1, p=1, b=bias),
                Conv2(c, k=3, s=1, p=1, b=bias)
            )

    def forward(self, x, cat_in=None):
        if cat_in is not None:
            x = self.net(torch.cat([x, cat_in], dim=1))
        else:
            x = self.net(x)
        printit(f'conv {x.shape}')
        return x

class Downsample(nn.Module):
    def __init__(self, dim=3):
        super(Downsample, self).__init__()
        if dim == 3:
            self.net = nn.MaxPool3d(kernel_size=2)
        else:
            self.net = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.net(x)
        printit(f'down {x.shape}')
        return x

class Upsample(nn.Module):
    def __init__(self, c, out_shape, dim=3, bias=False):
        super(Upsample, self).__init__()
        #self.net = TConv3(c, k=2, s=2, p=0)
        if dim == 3:
            conv = Conv3(c, k=3, s=1, p=1, b=bias)
        else:
            conv = Conv2(c, k=3, s=1, p=1, b=bias)

        self.net = nn.Sequential(
            nn.Upsample(size=out_shape, mode='nearest'),
            conv
        )

    def forward(self, x):
        x = self.net(x)
        printit(f'up {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, depth=16, init_c=32, dim=3, bias=False):
        super(UNet, self).__init__()
        self.dim = dim
        # down layers
        self.down = nn.ModuleList([
            ConvBlock(init_c, dim, bias), Downsample(dim), ConvBlock(2 * init_c, dim, bias), Downsample(dim), ConvBlock(4 * init_c, dim, bias),
            Downsample(dim), ConvBlock(8 * init_c, dim, bias), Downsample(dim), ConvBlock(16 * init_c, dim, bias)
        ])
        # up layers
        self.up = nn.ModuleList([
            Upsample(8 * init_c, (6, 5), dim, bias), ConvBlock(8 * init_c, dim, bias), Upsample(4 * init_c, (13, 10), dim, bias), ConvBlock(4 * init_c, dim, bias),
            Upsample(2 * init_c, (27, 21), dim, bias), ConvBlock(2 * init_c, dim, bias), Upsample(init_c, (54, 42), dim, bias), ConvBlock(init_c, dim, bias)
        ])
        if dim == 3:
            self.red_depth = Conv3(init_c, k=(depth, 1, 1), s=1, p=0, b=bias)
        self.compress = nn.Sequential(
            Conv2(init_c, k=3, s=1, p=1, b=bias),
            Conv2(init_c // 2, k=3, s=1, p=1, b=bias),
            #Conv2(N_CLASS, k=1, s=1, p='same', b=bias),  # none, light, moderate, extreme precip.? but then not end-to-end?
            nn.LazyConv2d(N_CLASS, kernel_size=1, stride=1, padding=0, bias=bias)
        )

        self.smax = nn.Identity()

    def forward(self, x):
        outs = []
        for i, step in enumerate(self.down):
            x = step(x)
            if i % 2 == 0 and i != len(self.down) - 1:
                outs.append(x)

        ctr = 0
        for j, step in enumerate(self.up):
            if (j + 1) % 2 == 0:
                ctr += 1
                x = step(x, cat_in=outs[-ctr])
            else:
                x = step(x)

        if self.dim == 3:
            x = self.red_depth(x).squeeze(2)
            printit(f'red depth {x.shape}')
        x = self.smax(self.compress(x))
        printit(f'final {x.shape}')
        return x

class MultiUNet(nn.Module):
    def __init__(self, n_vars=5, depth=16, spatial_dim=2, init_c=32, embedding_dim=32, bias=False):
        super(MultiUNet, self).__init__()
        self.unets = nn.ModuleList([UNet(depth=depth, init_c=init_c, dim=spatial_dim, bias=bias) for _ in range(n_vars)])
        self.final = nn.LazyConv2d(1, kernel_size=1, stride=1, padding=0)
        # train these on target first? and then train second level for generation from classifiers?
        #self.projection = Conv2(embedding_dim, k=1, s=1, p=0, b=bias)
        #self.projection = nn.Linear()
        # this is either a per-pixel transformer or 1d conv or just average.?
        # somehow need to have 'image generator'.?
        #self.concept = Conv2(N_CLASS, k=1, s=1, p=0, b=bias)

    def forward(self, x):
        outputs = []
        for d, net in enumerate(self.unets):
            outputs.append(net(x[:, d].squeeze(1)))

        final = torch.zeros_like(outputs[0])
        for out in outputs:
            final += out
        #final /= len(outputs)  # averaging for label fusion.?
        return F.sigmoid(self.final(final))
        #merge_x = torch.cat(outputs, dim=1)
        #project_x = self.projection(merge_x)
        #return F.softmax(self.concept(project_x), dim=1)

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

