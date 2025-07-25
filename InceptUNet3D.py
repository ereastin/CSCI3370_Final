import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from Components import *

PRINT = False
ROI = False
# ---------------------------------------------------------------------------------
def main():
    base = 32
    lin_act = 0.117
    Na, Nb, Nc = 5, 10, 5  #3, 6, 3
    lr = 2.44e-04
    wd = 0.036
    drop_p = 0.163
    bias = True

    # use this insead of k3 s2?
    #print(calc_shape_out(144, 0, 2, 37, 1))
    #print(calc_shape_out(80, 0, 2, 21, 1))
    #print(calc_shape_out(35, 0, 1, 18, 1))
    #exit()

    net = IRNv4_3DUNet(5, depth=16, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act).cuda()
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

# ---------------------------------------------------------------------------------
class S1(nn.Module):
    # C -> 2b (b == C as of now)
    def __init__(self, base, b=False, drop_p=0.0):
        super(S1, self).__init__()
        self.stem_a = nn.Sequential(
            Conv3(base, k=(3, 1, 1), s=1, p='same', d=3, b=b),
            Conv3(base, k=(1, 3, 1), s=1, p='same', d=5, b=b),
            Conv3(base, k=(1, 1, 3), s=1, p='same', d=5, b=b),
            Conv3(2 * base, k=3, s=1, p=0, b=b, drop_p=drop_p)
            #Conv3(2 * base, k=3, s=2, p=1, b=b, drop_p=drop_p),
            #Conv3(base, k=3, s=2, p=1, b=b),
            # TODO: not sure this above actually here.? my original impl didnt reduce here
            # Conv3(base, k=3, s=1, p=1, b=b),
            #Conv3(2 * base, k=3, s=1, p=1, b=b, drop_p=drop_p)
        )

    def forward(self, x):
        return self.stem_a(x)

# ---------------------------------------------------------------------------------
class S2(nn.Module):
    # 2b -> 5b
    def __init__(self, base, b=False, drop_p=0.0):
        super(S2, self).__init__()
        self.stem_b1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.stem_b2 = Conv3(3 * base, k=3, s=2, p=1, b=b)
        self.drop = nn.Dropout3d(p=drop_p)

    def forward(self, x):
        b1 = self.stem_b1(x)
        b2 = self.stem_b2(x)
        printit(f'S2 outputs: {b1.shape}, {b2.shape}')
        return self.drop(torch.cat([b1, b2], dim=1))

# ---------------------------------------------------------------------------------
class S3(nn.Module):
    # 5b -> 6b
    def __init__(self, base, b=False, drop_p=0.0):
        super(S3, self).__init__()
        self.stem_c1 = nn.Sequential(
            Conv3(2 * base, k=1, s=1, p='same', b=b),
            Conv3(3 * base, k=3, s=1, p='same', b=b)
        )
        self.stem_c2 = nn.Sequential(
            Conv3(2 * base, k=1, s=1, p='same', b=b),
            Conv3(2 * base, k=(1, 7, 1), s=1, p='same', b=b),
            Conv3(2 * base, k=(1, 1, 7), s=1, p='same', b=b),
            Conv3(2 * base, k=(7, 1, 1), s=1, p='same', b=b),
            Conv3(3 * base, k=3, s=1, p='same', b=b)
        )
        self.drop = nn.Dropout3d(p=drop_p)

    def forward(self, x):
        c1 = self.stem_c1(x)
        c2 = self.stem_c2(x)
        printit(f'S3 outputs: {c1.shape}, {c2.shape}')
        return self.drop(torch.cat([c1, c2], dim=1))

# ---------------------------------------------------------------------------------
class S4(nn.Module):
    # 6b -> 12b
    def __init__(self, base, b=False, drop_p=0.0):
        super(S4, self).__init__()
        self.stem_d1 = Conv3(6 * base, k=3, s=2, p=1, b=b)
        self.stem_d2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout3d(p=drop_p)

    def forward(self, x):
        d1 = self.stem_d1(x)
        d2 = self.stem_d2(x)
        printit(f'S4 outputs: {d1.shape}, {d2.shape}')
        return self.drop(torch.cat([d1, d2], dim=1))

# ---------------------------------------------------------------------------------
class A(nn.Module):
    # 12b -> 12b, reverse (12 + 12)b -> 12b
    def __init__(self, base, b=False, drop_p=0.0, lin_act=0.1, reverse=False):
        super(A, self).__init__()
        self.lin_act = lin_act
        self.reverse = reverse
        scale = 2 if self.reverse else 1
        self.a1 = Conv3(base * scale, k=1, s=1, p='same', b=b)
        self.a2 = nn.Sequential(
            Conv3(base * scale, k=1, s=1, p='same', b=b),
            Conv3(base * scale, k=3, s=1, p='same', b=b)
        )
        self.a3 = nn.Sequential(
            Conv3(base * scale, k=1, s=1, p='same', b=b),
            Conv3(3 * base * scale // 2, k=3, s=1, p='same', b=b),
            Conv3(2 * base * scale, k=3, s=1, p='same', b=b)
        )
        self.comb = nn.LazyConv3d(12 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x, cat_in=None):
        if self.reverse: x = torch.cat([x, cat_in], dim=1)
        a1 = self.a1(x)
        a2 = self.a2(x)
        a3 = self.a3(x)
        printit(f'A outputs: {a1.shape}, {a2.shape}, {a3.shape}')
        x_conv = torch.cat([a1, a2, a3], dim=1)
        x_conv = self.lin_act * self.comb(x_conv) # linear activation scaling for stability
        return self.final(x + x_conv)  # residual connection

# ---------------------------------------------------------------------------------
class redA(nn.Module):
    # 12b -> 24b, traditionally this triples channel depth.. necessary?
    def __init__(self, base, b=False, drop_p=0.0):
        super(redA, self).__init__()
        self.ra1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.ra2 = Conv3(6 * base, k=3, s=2, p=1, b=b)
        self.ra3 = nn.Sequential(
            Conv3(4 * base, k=1, s=1, p='same', b=b),
            Conv3(4 * base, k=3, s=1, p='same', b=b),
            Conv3(6 * base, k=3, s=2, p=1, b=b)
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        ra1 = self.ra1(x)
        ra2 = self.ra2(x)
        ra3 = self.ra3(x)
        printit(f'Red A outputs: {ra1.shape}, {ra2.shape}, {ra3.shape}')
        return self.final(torch.cat([ra1, ra2, ra3], dim=1))

# ---------------------------------------------------------------------------------
class B(nn.Module):
    # 24b -> 24b, reverse (24 + 24)b -> 24b
    def __init__(self, base, b=False, drop_p=0.0, lin_act=0.1):
        super(B, self).__init__()
        self.lin_act = lin_act
        self.b1 = Conv3(6 * base, k=1, s=1, p='same', b=b)
        self.b2 = nn.Sequential(
            Conv3(4 * base, k=1, s=1, p='same', b=b),
            Conv3(5 * base, k=(1, 7, 1), s=1, p='same', b=b),
            Conv3(6 * base, k=(1, 1, 7), s=1, p='same', b=b),
            Conv3(7 * base, k=(7, 1, 1), s=1, p='same', b=b)  # consider this?
        )
        self.comb = nn.LazyConv3d(24 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        printit(f'B outputs: {b1.shape}, {b2.shape}')
        x_conv = torch.cat([b1, b2], dim=1)
        x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
        return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class redB(nn.Module):
    # 24b -> 48b
    def __init__(self, base, b=False, drop_p=0.0):
        super(redB, self).__init__()
        self.rb1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.rb2 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b),
            Conv3(7 * base, k=3, s=2, p=1, b=b)
        )
        self.rb3 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b),
            Conv3(8 * base, k=3, s=2, p=1, b=b)
        )
        self.rb4 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b),
            Conv3(7 * base, k=3, s=1, p='same', b=b),
            Conv3(9 * base, k=3, s=2, p=1, b=b)
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        rb1 = self.rb1(x)
        rb2 = self.rb2(x)
        rb3 = self.rb3(x)
        rb4 = self.rb4(x)
        printit(f'Red B outputs: {rb1.shape}, {rb2.shape}, {rb3.shape}, {rb4.shape}')
        return self.final(torch.cat([rb1, rb2, rb3, rb4], dim=1))

# ---------------------------------------------------------------------------------
class C(nn.Module):
    # 48b -> 48b
    def __init__(self, base, b=False, drop_p=0.0, lin_act=0.1):
        super(C, self).__init__()
        self.lin_act = lin_act
        self.c1 = Conv3(6 * base, k=1, s=1, p=0, b=b)
        self.c2 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b),
            Conv3(7 * base, k=(1, 1, 3), s=1, p='same', b=b),
            Conv3(8 * base, k=(1, 3, 1), s=1, p='same', b=b),
            Conv3(9 * base, k=(3, 1, 1), s=1, p='same', b=b),  # consider
        )
        self.comb = nn.LazyConv3d(48 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(x)
        printit(f'C outputs: {c1.shape}, {c2.shape}')
        x_conv = torch.cat([c1, c2], dim=1)
        x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
        return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class revB(nn.Module):
    # 24b -> 24b, reverse (24 + 24)b -> 24b
    def __init__(self, base, b=False, drop_p=-1.0, lin_act=0.1):
        super(revB, self).__init__()
        self.lin_act = lin_act
        scale = 3
        self.b1 = Conv2(6 * base * scale, k=1, s=1, p='same', b=b)
        self.b2 = nn.Sequential(
            Conv2(7 * base * scale, k=1, s=1, p='same', b=b),
            Conv2(5 * base * scale, k=(1, 7), s=1, p='same', b=b),
            Conv2(4 * base * scale, k=(7, 1), s=1, p='same', b=b)
        )
        self.comb = nn.LazyConv2d(24 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.compress = nn.Sequential(
            Conv3(4 * base, k=(2, 1, 1), s=1, p=0, b=b),
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x, cat_in):
        printit(f'revB in: {x.shape}, {cat_in.shape}')
        cat_in = self.compress(cat_in).squeeze(2)
        printit(f'revb comp {cat_in.shape}')
        x_cat = torch.cat([x, cat_in], dim=1)
        b1 = self.b1(x_cat)
        b2 = self.b2(x_cat)
        printit(f'revB outputs: {b1.shape}, {b2.shape}')
        x_conv = torch.cat([b1, b2], dim=1)
        x_conv = self.lin_act * self.comb(x_conv)  # linear activation scaling for stability
        return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class revA(nn.Module):
    # 12b -> 12b, reverse (12 + 12)b -> 12b
    def __init__(self, base, b=False, drop_p=0.0, lin_act=0.1, reverse=False):
        super(revA, self).__init__()
        self.lin_act = lin_act
        scale = 3
        self.a1 = Conv2(base, k=1, s=1, p='same', b=b)
        self.a2 = nn.Sequential(
            Conv2(base * scale, k=1, s=1, p='same', b=b),
            Conv2(base * scale, k=3, s=1, p='same', b=b)
        )
        self.a3 = nn.Sequential(
            Conv2(2 * base * scale, k=1, s=1, p='same', b=b),
            Conv2(3 * base * scale // 2, k=3, s=1, p='same', b=b),
            Conv2(base * scale, k=3, s=1, p='same', b=b)
        )
        self.comb = nn.LazyConv2d(12 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.compress = nn.Sequential(
            Conv3(2 * base, k=(4, 1, 1), s=1, p=0, b=b),
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x, cat_in):
        printit(f'revA in: {x.shape}, {cat_in.shape}')
        cat_in = self.compress(cat_in).squeeze(2)
        printit(f'reva comp {cat_in.shape}')
        x_cat = torch.cat([x, cat_in], dim=1)
        a1 = self.a1(x_cat)
        a2 = self.a2(x_cat)
        a3 = self.a3(x_cat)
        printit(f'revA outputs: {a1.shape}, {a2.shape}, {a3.shape}')
        x_conv = torch.cat([a1, a2, a3], dim=1)
        x_conv = self.lin_act * self.comb(x_conv) # linear activation scaling for stability
        return self.final(x + x_conv)  # residual connection

# ---------------------------------------------------------------------------------
"""
class revB(nn.Module):
    # NOTE: think just use A == revA, B == revB
    def __init__(self, in_size, out_size, b=False, drop_p=0.0):
        super(revB, self).__init__()
        self.rb1 = TConv2(in_size // 5, k=1, s=1, p=0, b=b, res=True) 
        self.rb2 = nn.Sequential(
            TConv2(in_size // 9, k=(7, 1), s=1, p=(3, 0), b=b, res=True),
            TConv2(in_size // 8, k=(1, 7), s=1, p=(0, 3), b=b, res=True),
            TConv2(in_size // 7, k=1, s=1, p=0, b=b, res=True)
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout2d(p=drop_p)
        )
        self.trim = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

    def forward(self, x, cat_in):
        printit(f'Rev B in and cat in: {x.shape}, {cat_in.shape}')
        # this still work?
        cat_in = self.trim(cat_in)
        printit("CAT", cat_in.shape)
        comb = torch.cat([x, cat_in], dim=1)
        rb1 = self.rb1(comb)
        rb2 = self.rb2(comb)
        printit(f'Rev B outputs: {rb1.shape}, {rb2.shape}')
        return self.final(torch.cat([rb1, rb2], dim=1))
"""

# ---------------------------------------------------------------------------------
"""
class revA(nn.Module):
    # NOTE: think just use A == revA, B == revB
    def __init__(self, in_size, out_size, b=False, drop_p=0.0):
        super(revA, self).__init__()
        self.ra1 = Conv2(in_size, k=1, s=1, p=0, b=b)
        self.ra2 = nn.Sequential(
            Conv2(in_size, k=3, s=1, p=1, b=b, res=True),
            Conv2(in_size, k=1, s=1, p=0, b=b, res=True)
        )
        self.ra3 = nn.Sequential(
            TConv2(in_size // 7, k=3, s=1, p=1, b=b, res=True),
            TConv2(in_size // 6, k=3, s=1, p=1, b=b, res=True),
            TConv2(in_size // 5, k=1, s=1, p=0, b=b, res=True)
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x, cat_in):
        printit(f'Rev A in and cat in: {x.shape}, {cat_in.shape}')
        s = cat_in.shape
        comb = torch.cat([x, cat_in], dim=1)
        ra1 = self.ra1(comb)
        ra2 = self.ra2(comb)
        ra3 = self.ra3(comb)
        printit(f'Rev A outputs: {ra1.shape}, {ra2.shape}, {ra3.shape}')
        return self.final(torch.cat([ra1, ra2, ra3], dim=1))

"""
# ---------------------------------------------------------------------------------
class revRA(nn.Module):
    # 24b -> 12b
    def __init__(self, base, op=1, b=False, drop_p=0.0):
        super(revRA, self).__init__()
        # self.rra1 = nn.Upsample(size=(10, 18))  # if this module is reduce channels by half cant use this..
        self.rra2 = TConv2(6 * base, k=3, s=2, p=1, op=1, b=b)
        self.rra3 = nn.Sequential(
            Conv2(4 * base, k=1, s=1, p=0, b=b),
            Conv2(4 * base, k=3, s=1, p=1, b=b, res=True),
            TConv2(6 * base, k=3, s=2, p=1, op=1, b=b),
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        #rra1 = self.rra1(x)
        rra2 = self.rra2(x)
        rra3 = self.rra3(x)
        printit(f'Rev redA outputs: {rra2.shape}, {rra3.shape}')
        return self.final(torch.cat([rra2, rra3], dim=1))

# ---------------------------------------------------------------------------------
class revRB(nn.Module):
    # 48b -> 24b
    def __init__(self, base, b=False, drop_p=0.0):
        super(revRB, self).__init__()
        # self.rrb1 = nn.Upsample(scale_factor=2)  # just remove the upsample.?
        self.rrb2 = nn.Sequential(
            Conv2(10 * base, k=1, s=1, p=0, b=b),
            TConv2(7 * base, k=3, s=2, p=1, op=1, b=b),
        )
        self.rrb3 = nn.Sequential(
            Conv2(10 * base, k=1, s=1, p=0, b=b),
            TConv2(8 * base, k=3, s=2, p=1, op=1, b=b),
        )
        self.rrb4 = nn.Sequential(
            Conv2(11 * base, k=3, s=1, p=1, b=b),
            Conv2(10 * base, k=1, s=1, p=0, b=b),
            TConv2(9 * base, k=3, s=2, p=1, op=1, b=b),
        )
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        #rrb1 = self.rrb1(x)
        rrb2 = self.rrb2(x)
        rrb3 = self.rrb3(x)
        rrb4 = self.rrb4(x)
        printit(f'Rev redB outputs: {rrb2.shape}, {rrb3.shape}, {rrb4.shape}')
        return self.final(torch.cat([rrb2, rrb3, rrb4], dim=1))

# ---------------------------------------------------------------------------------
class revS1(nn.Module):
    # (2 + 2)b -> 2b
    def __init__(self, base, b=False, drop_p=0.0):
        super(revS1, self).__init__()
        self.rs1 = nn.Sequential(
            Conv2(2 * base, k=3, s=1, p=1, b=b),
            Conv2(2 * base, k=3, s=1, p=1, b=b, res=True),
            TConv2(base, k=3, s=1, p=0, op=0, b=b, drop_p=drop_p)
        )
        self.compress = nn.Sequential(
            Conv3(2 * base, k=(14, 1, 1), s=1, p=0, b=b),
        )

    def forward(self, x, cat_in):
        printit(f'revS1: {x.shape, cat_in.shape}')
        cat_in = self.compress(cat_in).squeeze(2)
        printit(f'revs1 comp {cat_in.shape}')
        cat = torch.cat([x, cat_in], dim=1)
        return self.rs1(cat)

# ---------------------------------------------------------------------------------
class revS1_ROI(nn.Module):
    # (2 + 2)b -> 2b
    def __init__(self, base, b=False, drop_p=0.0):
        super(revS1_ROI, self).__init__()
        self.rs1 = nn.Sequential(
            Conv2(2 * base, k=(3, 5), s=1, p='same', b=b),
            Conv2(2 * base, k=(5, 3), s=1, p='same', b=b, drop_p=drop_p, res=True),
        )
        self.rs2 = nn.Sequential(
            Conv2(2 * base, k=(3, 5), s=1, p='same', b=b),
            Conv2(2 * base, k=(5, 3), s=1, p='same', b=b, drop_p=drop_p, res=True),
        )
        self.rs3 = nn.Sequential(
            Conv2(2 * base, k=3, s=1, p=0, b=b),
            Conv2(2 * base, k=3, s=1, p=0, b=b),
        )
        self.compress = nn.Sequential(
            Conv3(2 * base, k=(14, 1, 1), s=1, p=0, b=b),
        )

    def forward(self, x, cat_in):
        cat_in = self.compress(cat_in)
        printit(f'revs1 comp {cat_in.shape}')
        cat = torch.cat([x, cat_in], dim=1)
        s = cat.shape
        cat = cat.reshape(s[0], s[1], 60, 48)  # dont love the hard coded or the reshape this far down..
        out = self.rs1(cat)
        out = self.rs2(out)
        return self.rs3(out)

# ---------------------------------------------------------------------------------
class revS2(nn.Module):
    # 5b -> 2b
    def __init__(self, base, b=False, drop_p=0.0):
        super(revS2, self).__init__()
        #self.rs2a = nn.Upsample(scale_factor=2)
        self.rs2b = TConv2(2 * base, k=3, s=2, p=1, op=1, b=b)
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        #rs2a = self.rs2a(x)
        rs2b = self.rs2b(x)
        #printit(f'Rev S2 out: {rs2a.shape}, {rs2b.shape}')
        #return self.final(torch.cat([rs2a, rs2b], dim=1))
        printit(f'Rev S2 out: {rs2b.shape}')
        return self.final(rs2b)

# ---------------------------------------------------------------------------------
class revS3(nn.Module):
    # (6 + 6)b -> 5b
    def __init__(self, base, b=False, drop_p=0.0):
        super(revS3, self).__init__()
        self.rs3a = nn.Sequential(
            Conv2(3 * base, k=3, s=1, p=1, b=b),
            Conv2(2 * base, k=1, s=1, p=0, b=b, drop_p=drop_p)
        )
        self.rs3b = nn.Sequential(
            Conv2(3 * base, k=3, s=1, p=1, b=b),
            Conv2(3 * base, k=(1, 7), s=1, p=(0, 3), b=b, res=True),
            Conv2(3 * base, k=(7, 1), s=1, p=(3, 0), b=b, res=True),
            Conv2(3 * base, k=1, s=1, p=0, b=b, drop_p=drop_p, res=True)
        )
        self.compress = nn.Sequential(
            Conv3(3 * base, k=(7, 1, 1), s=1, p=0, b=b),
        )

    def forward(self, x, cat_in):
        printit(f'revS3: {x.shape, cat_in.shape}')
        cat_in = self.compress(cat_in).squeeze(2)
        printit(f'revs3 comp {cat_in.shape}')
        cat = torch.cat([x, cat_in], dim=1)
        rs3a = self.rs3a(cat)
        rs3b = self.rs3b(cat)
        printit(f'Rev S3 out: {rs3a.shape}, {rs3b.shape}')
        return torch.cat([rs3a, rs3b], dim=1)

# ---------------------------------------------------------------------------------
class revS4(nn.Module):
    # 12b -> 6b
    def __init__(self, base, b=False, drop_p=0.0):
        super(revS4, self).__init__()
        #self.rs4a = nn.Upsample(scale_factor=2)  # wtf should i do for these
        self.rs4b = TConv2(6 * base, k=3, s=2, p=1, op=0, b=b)
        self.final = nn.Sequential(
            nn.SELU(),
            nn.Dropout2d(p=drop_p)
        )

    def forward(self, x):
        #rs4a = self.rs4a(x)
        rs4b = self.rs4b(x)
        printit(f'Rev S4 outputs: {rs4b.shape}')
        #return self.final(torch.cat([rs4a, rs4b], dim=1))
        return self.final(rs4b)

# ---------------------------------------------------------------------------------
class IRNv4_3DUNet(nn.Module):
    def __init__(self, in_size, depth=42, Na=1, Nb=1, Nc=1, base=32, bias=False, drop_p=0.0, lin_act=0.1):
        """
        Na, Nb, Nc: number of times to run through layers A, B, C
        Inception-ResNet-v1/2 use 5, 10, 5
        """
        super(IRNv4_3DUNet, self).__init__()
        self.depth = depth
        # down layers
        self.s1 = S1(base, b=bias, drop_p=drop_p)  # -> (B, 64, , H, W)
        self.s2 = S2(base, b=bias, drop_p=drop_p)  # -> (B, 160, , H, W)
        self.s3 = S3(base, b=bias, drop_p=drop_p)  # -> (B, 192, , H, W)
        self.s4 = S4(base, b=bias, drop_p=drop_p)  # -> (B, 384, D, H, W)
        self.a_n = [A(base, b=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Na)]  # (B, 384, D, H, W)
        self.ra = redA(base, b=bias, drop_p=drop_p)  # -> (B, 1152, D, H, W) this layer TRIPLES C
        self.b_n = [B(base, b=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Nb)]  # -> (B, 1152, D, H, W)
        self.rb = redB(base, b=bias, drop_p=drop_p)  # (B, 2048, 1, H, W)
        self.c_n = [C(base, b=bias, drop_p=drop_p, lin_act=lin_act) for _ in range(Nc)]  # -> (B, 2048, 1, H, W)

        # up layers
        self.rev_rb = revRB(base, b=bias, drop_p=drop_p)  # (B, 1152, ...
        self.rev_b = revB(base, b=bias, drop_p=drop_p)  # (B, 1152, ... CAT
        self.rev_ra = revRA(base, b=bias, drop_p=drop_p)  # (B, 384, ...
        self.rev_a = revA(base, b=bias, drop_p=drop_p)  # (B, 384, ... CAT
        self.rev_s4 = revS4(base, b=bias, drop_p=drop_p)  # (B, 192, ...)
        self.rev_s3 = revS3(base, b=bias, drop_p=drop_p)  # (B, 160, ...)  CAT
        self.rev_s2 = revS2(base, b=bias, drop_p=drop_p)  # (B, 128, ...)
        if not ROI:
            self.rev_s1 = revS1(base, b=bias, drop_p=drop_p)  # (B, 5, ...)  CAT
        else:
            self.rev_s1 = revS1_ROI(base, b=bias, drop_p=drop_p)

        self.final = nn.LazyConv2d(1, kernel_size=1, stride=1, padding=0, bias=bias)

        self.net_down = nn.ModuleList(
                [self.s1, self.s2, self.s3, self.s4] + self.a_n + [self.ra] + self.b_n + [self.rb] + self.c_n
        )

        self.net_up = nn.ModuleList([
            self.rev_rb, self.rev_b, self.rev_ra, self.rev_a,
            self.rev_s4, self.rev_s3, self.rev_s2, self.rev_s1
        ])
        self.cat_select = [0, 2, 3 + Na, 3 + Na + 1 + Nb]

    def forward(self, x):
        outs = []
        for i, step in enumerate(self.net_down):
            x = step(x)
            if i in self.cat_select:
                outs.append(x)
                printit(step, x.shape)

        # reshape to 2D in/out
        s = x.shape
        x = x.reshape(s[0], -1, s[3], s[4])
        #if ROI: x = self.bottom(x)
        printit(f'BOTTOM: {x.shape}')

        ctr = 0
        for j, step in enumerate(self.net_up):
            if (j + 1) % 2 == 0:  # (j + 1) % 2 == 0: is original
                ctr += 1
                printit(step, x.shape, outs[-ctr].shape)
                # possible to just reshape to cat shape on the fly?
                # e.g.
                # xs = x.shape
                # tmp = outs[-ctr]
                # tmp = tmp.reshape(xs)  ? this works assuming HtxWt = HxxWx..
                x = step(x, outs[-ctr])
            else:
                x = step(x)
                printit(step, x.shape)

        printit('FINAL', x.shape)
        x = self.final(x)
        return x

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

