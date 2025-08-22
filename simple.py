import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from Components import *

PRINT = False
ROI = True
ATTN = True
N, D, H, W = 6, 28, 81, 145
# ---------------------------------------------------------------------------------
def main():
    base = 36
    lin_act = 0.117
    Na, Nb, Nc = 5, 10, 5
    lr = 2.44e-04
    wd = 0.036
    drop_p = 0.163
    bias = True

    # use this insead of k3 s2?
    #print(calc_shape_out(144, 0, 2, 37, 1))
    #print(calc_shape_out(80, 0, 2, 21, 1))
    #print(calc_shape_out(35, 0, 1, 18, 1))
    #exit()

    net = Simple(N, depth=D, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act).cuda()
    #print(net)
    test = torch.ones((1, N, D, H, W)).cuda()
    net(test)
    #print(net)  doing 54, 42 is just too small with all the downsampling..
    s = (64, N, D, H, W)
    summary(net, input_size=s)

def printit(*args):
    print(*args) if PRINT else None

# ---------------------------------------------------------------------------------
class S1(nn.Module):
    # C -> 2b
    def __init__(self, base, b=False, drop_p=0.0):
        super(S1, self).__init__()
        self.s1_b1 = nn.Sequential(
            Conv3(base, k=(3, 1, 1), s=1, p='same', d=1, b=b, drop_p=drop_p),
            Conv3(base, k=(1, 3, 1), s=1, p='same', d=1, b=b, drop_p=drop_p),
            Conv3(base, k=(1, 1, 3), s=1, p='same', d=1, b=b, drop_p=drop_p),
            Conv3(2 * base, k=3, s=2, p=0, b=b, drop_p=drop_p)
        )

    def forward(self, x):
        b1 = self.s1_b1(x)
        printit(f'S1 outputs: {b1.shape}')
        return b1

# ---------------------------------------------------------------------------------
class S2(nn.Module):
    # 2b -> 5b
    def __init__(self, base, b=False, drop_p=0.0):
        super(S2, self).__init__()
        self.s2_b1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.s2_b2 = Conv3(3 * base, k=3, s=2, p=1, b=b, drop_p=drop_p)

    def forward(self, x):
        b1 = self.s2_b1(x)
        b2 = self.s2_b2(x)
        printit(f'S2 outputs: {b1.shape}, {b2.shape}')
        return torch.cat([b1, b2], dim=1)

# ---------------------------------------------------------------------------------
class S3(nn.Module):
    # 5b -> 6b
    def __init__(self, base, b=False, drop_p=0.0):
        super(S3, self).__init__()
        self.s3_b1 = nn.Sequential(
            Conv3(2 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(3 * base, k=3, s=1, p='same', b=b, drop_p=drop_p)
        )
        self.s3_b2 = nn.Sequential(
            Conv3(2 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(2 * base, k=(1, 7, 1), s=1, p='same', b=b, drop_p=drop_p),
            Conv3(2 * base, k=(1, 1, 7), s=1, p='same', b=b, drop_p=drop_p),
            Conv3(2 * base, k=(7, 1, 1), s=1, p='same', b=b, drop_p=drop_p),
            Conv3(3 * base, k=3, s=1, p='same', b=b, drop_p=drop_p)
        )

    def forward(self, x):
        b1 = self.s3_b1(x)
        b2 = self.s3_b2(x)
        printit(f'S3 outputs: {b1.shape}, {b2.shape}')
        return torch.cat([b1, b2], dim=1)

# ---------------------------------------------------------------------------------
class S4(nn.Module):
    # 6b -> 12b
    def __init__(self, base, b=False, drop_p=0.0):
        super(S4, self).__init__()
        self.s4_b1 = Conv3(6 * base, k=3, s=2, p=1, b=b, drop_p=drop_p)
        self.s4_b2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        b1 = self.s4_b1(x)
        b2 = self.s4_b2(x)
        printit(f'S4 outputs: {b1.shape}, {b2.shape}')
        return torch.cat([b1, b2], dim=1)

# ---------------------------------------------------------------------------------
class A(nn.Module):
    # 12b -> 12b
    def __init__(self, base, b=False, drop_p=0.0, lin_act=0.1):
        super(A, self).__init__()
        self.lin_act = lin_act
        self.a_b1 = Conv3(base, k=1, s=1, p='same', b=b, drop_p=drop_p)
        self.a_b2 = nn.Sequential(
            Conv3(base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(base, k=3, s=1, p='same', b=b, drop_p=drop_p)
        )
        self.a_b3 = nn.Sequential(
            Conv3(base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(3 * base // 2, k=3, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(2 * base, k=3, s=1, p='same', b=b, drop_p=drop_p)
        )
        self.combine = nn.LazyConv3d(12 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.final = nn.ReLU()

    def forward(self, x):
        b1 = self.a_b1(x)
        b2 = self.a_b2(x)
        b3 = self.a_b3(x)
        printit(f'A outputs: {b1.shape}, {b2.shape}, {b3.shape}')
        x_conv = torch.cat([b1, b2, b3], dim=1)
        x_conv = self.lin_act * self.combine(x_conv) # linear activation scaling for stability
        return self.final(x + x_conv)  # residual connection .. want relu/dropout here?

# ---------------------------------------------------------------------------------
class redA(nn.Module):
    # 12b -> 24b, traditionally this triples channel depth.. necessary?
    def __init__(self, base, b=False, drop_p=0.0):
        super(redA, self).__init__()
        self.ra_b1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.ra_b2 = Conv3(6 * base, k=3, s=2, p=1, b=b, drop_p=drop_p)
        self.ra_b3 = nn.Sequential(
            Conv3(4 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(4 * base, k=3, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(6 * base, k=3, s=2, p=1, b=b, drop_p=drop_p)
        )

    def forward(self, x):
        b1 = self.ra_b1(x)
        b2 = self.ra_b2(x)
        b3 = self.ra_b3(x)
        printit(f'Red A outputs: {b1.shape}, {b2.shape}, {b3.shape}')
        return torch.cat([b1, b2, b3], dim=1)

# ---------------------------------------------------------------------------------
class B(nn.Module):
    # 24b -> 24b
    def __init__(self, base, b=False, drop_p=0.0, lin_act=0.1):
        super(B, self).__init__()
        self.lin_act = lin_act
        self.b_b1 = Conv3(6 * base, k=1, s=1, p='same', b=b, drop_p=drop_p)
        self.b_b2 = nn.Sequential(
            Conv3(4 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(5 * base, k=(1, 7, 1), s=1, p='same', b=b, drop_p=drop_p),
            Conv3(6 * base, k=(1, 1, 7), s=1, p='same', b=b, drop_p=drop_p),
            Conv3(7 * base, k=(7, 1, 1), s=1, p='same', b=b, drop_p=drop_p)  # consider this?
        )
        self.combine = nn.LazyConv3d(24 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        b1 = self.b_b1(x)
        b2 = self.b_b2(x)
        printit(f'B outputs: {b1.shape}, {b2.shape}')
        x_conv = torch.cat([b1, b2], dim=1)
        x_conv = self.lin_act * self.combine(x_conv)  # linear activation scaling for stability
        return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class redB(nn.Module):
    # 24b -> 48b
    def __init__(self, base, b=False, drop_p=0.0):
        super(redB, self).__init__()
        self.rb_b1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.rb_b2 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(7 * base, k=3, s=2, p=1, b=b, drop_p=drop_p)
        )
        self.rb_b3 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(8 * base, k=3, s=2, p=1, b=b, drop_p=drop_p)
        )
        self.rb_b4 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(7 * base, k=3, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(9 * base, k=3, s=2, p=1, b=b, drop_p=drop_p)
        )

    def forward(self, x):
        b1 = self.rb_b1(x)
        b2 = self.rb_b2(x)
        b3 = self.rb_b3(x)
        b4 = self.rb_b4(x)
        printit(f'Red B outputs: {b1.shape}, {b2.shape}, {b3.shape}, {b4.shape}')
        return torch.cat([b1, b2, b3, b4], dim=1)

# ---------------------------------------------------------------------------------
class C(nn.Module):
    # 48b -> 48b
    def __init__(self, base, b=False, drop_p=0.0, lin_act=0.1):
        super(C, self).__init__()
        self.lin_act = lin_act
        self.c_b1 = Conv3(6 * base, k=1, s=1, p=0, b=b, drop_p=drop_p)
        self.c_b2 = nn.Sequential(
            Conv3(6 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv3(7 * base, k=(1, 1, 3), s=1, p='same', b=b, drop_p=drop_p),
            Conv3(8 * base, k=(1, 3, 1), s=1, p='same', b=b, drop_p=drop_p),
            Conv3(9 * base, k=(3, 1, 1), s=1, p='same', b=b, drop_p=drop_p),  # consider
        )
        self.combine = nn.LazyConv3d(48 * base, kernel_size=1, stride=1, padding=0, bias=b)
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x):
        b1 = self.c_b1(x)
        b2 = self.c_b2(x)
        printit(f'C outputs: {b1.shape}, {b2.shape}')
        x_conv = torch.cat([b1, b2], dim=1)
        x_conv = self.lin_act * self.combine(x_conv)  # linear activation scaling for stability
        return self.final(x + x_conv)

# ---------------------------------------------------------------------------------
class invB(nn.Module):
    # (24 + 24)b -> 24b
    def __init__(self, base, in_depth, b=False, drop_p=0.0, attn=False):
        super(invB, self).__init__()
        self.attn = attn
        self.net = nn.Sequential(
            Conv2(36 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv2(24 * base, k=(1, 7), s=1, p='same', b=b, drop_p=drop_p),
            Conv2(24 * base, k=(7, 1), s=1, p='same', b=b, drop_p=drop_p)
        )

        if attn:
            self.compress = Attn(36 * base, 24 * base, in_depth, b=b)
        else:
            self.compress = Conv3(24 * base, k=(2, 1, 1), s=1, p=0, b=b, drop_p=drop_p) if D == 16 else Conv3(24 * base, k=1, s=1, p=0, b=b)

    def forward(self, x, cat_in):
        printit(f'invB in: {x.shape}, {cat_in.shape}')

        if self.attn:
            cat_in = self.compress(x, cat_in)
        else:
            cat_in = self.compress(cat_in).squeeze(2)
        printit(f'invB compress out {cat_in.shape}')

        x_cat = torch.cat([x, cat_in], dim=1)
        x = self.net(x_cat)
        printit(f'invB outputs: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class invA(nn.Module):
    # (12 + 12)b -> 12b
    def __init__(self, base, in_depth, b=False, drop_p=0.0, attn=False):
        super(invA, self).__init__()
        self.attn = attn
        self.net = nn.Sequential(
            Conv2(20 * base, k=1, s=1, p='same', b=b, drop_p=drop_p),
            Conv2(16 * base, k=3, s=1, p='same', b=b, drop_p=drop_p),
            Conv2(12 * base, k=3, s=1, p='same', b=b, drop_p=drop_p)
        )

        if attn:
            self.compress = Attn(16 * base, 12 * base, in_depth, b=b)
        else:
            self.compress = Conv3(12 * base, k=(4, 1, 1), s=1, p=0, b=b, drop_p=drop_p) if D == 16 else Conv3(12 * base, k=(2, 1, 1), s=1, p=0, b=b)

    def forward(self, x, cat_in):
        printit(f'invA in: {x.shape}, {cat_in.shape}')
        if self.attn:
            cat_in = self.compress(x, cat_in)
        else:
            cat_in = self.compress(cat_in).squeeze(2)
        printit(f'invA compress out {cat_in.shape}')
        x_cat = torch.cat([x, cat_in], dim=1)
        x = self.net(x_cat)
        printit(f'invA outputs: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class invRedA(nn.Module):
    # 24b -> 12b
    def __init__(self, base, b=False, drop_p=0.0):
        super(invRedA, self).__init__()
        self.net = Upsample2(12 * base, b=b)

    def forward(self, x, out_shape):
        x = self.net(x, out_shape)
        printit(f'invRedA output: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class invRedB(nn.Module):
    # 48b -> 24b
    def __init__(self, base, b=False, drop_p=0.0):
        super(invRedB, self).__init__()
        self.net = Upsample2(24 * base, b=b)

    def forward(self, x, out_shape):
        x = self.net(x, out_shape)
        printit(f'Rev redB output: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class invS1(nn.Module):
    # (2 + 2)b -> 2b
    def __init__(self, base, in_depth, b=False, drop_p=0.0, attn=False):
        super(invS1, self).__init__()
        self.attn = attn
        self.net = Upsample2(2 * base, b=b)  # TODO: this needs to change based on if S1 reduces size or not
        if attn:
            self.compress = Attn(4 * base, 2 * base, in_depth, b=b)
        else:
            self.compress = Conv3(2 * base, k=(14, 1, 1), s=1, p=0, b=b, drop_p=drop_p) if D == 16 else Conv3(2 * base, k=(8, 1, 1), s=1, p=0, b=b)

    def forward(self, x, cat_in, out_shape):
        printit(f'invS1: {x.shape, cat_in.shape}')
        if self.attn:
            cat_in = self.compress(x, cat_in)
        else:
            cat_in = self.compress(cat_in).squeeze(2)
        printit(f'invS1 compress out {cat_in.shape}')

        cat = torch.cat([x, cat_in], dim=1)
        x = self.net(cat, out_shape)
        printit(f'invS1 output: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class invS2(nn.Module):
    # 5b -> 2b
    def __init__(self, base, b=False, drop_p=0.0):
        super(invS2, self).__init__()
        self.net = Upsample2(2 * base, b=b)

    def forward(self, x, out_shape):
        x = self.net(x, out_shape)
        printit(f'invS2 out: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class invS3(nn.Module):
    # (6 + 6)b -> 5b
    def __init__(self, base, in_depth, b=False, drop_p=0.0, attn=False):
        super(invS3, self).__init__()
        self.attn = attn
        self.net = nn.Sequential(
            Conv2(10 * base, k=3, s=1, p=1, b=b, drop_p=drop_p),
            Conv2(8 * base, k=(1, 7), s=1, p=(0, 3), b=b, drop_p=drop_p),
            Conv2(6 * base, k=(7, 1), s=1, p=(3, 0), b=b, drop_p=drop_p),
            Conv2(5 * base, k=1, s=1, p=0, b=b, drop_p=drop_p)
        )
        if attn:
            self.compress = Attn(12 * base, 6 * base, in_depth, b=b)
        else:
            self.compress = nn.Sequential(Conv3(6 * base, k=(7, 1, 1), s=1, p=0, b=b, drop_p=drop_p) if D == 16 else Conv3(6 * base, k=(4, 1, 1), s=1, p=0, b=b))

    def forward(self, x, cat_in):
        printit(f'invS3 input: {x.shape, cat_in.shape}')
        if self.attn:
            cat_in = self.compress(x, cat_in)
        else:
            cat_in = self.compress(cat_in).squeeze(2)
        printit(f'invS3 compress out: {cat_in.shape}')

        cat = torch.cat([x, cat_in], dim=1)
        x = self.net(cat)
        printit(f'invS3 out: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class invS4(nn.Module):
    # 12b -> 6b
    def __init__(self, base, b=False, drop_p=0.0):
        super(invS4, self).__init__()
        self.net = Upsample2(6 * base, b=b)

    def forward(self, x, out_shape):
        x = self.net(x, out_shape)
        printit(f'invS4 output: {x.shape}')
        return x

# ---------------------------------------------------------------------------------
class Simple(nn.Module):
    def __init__(self, in_size, depth=42, Na=1, Nb=1, Nc=1, base=32, bias=False, drop_p=0.0, lin_act=0.1, attn=False):
        """
        Na, Nb, Nc: number of times to run through layers A, B, C
        Inception-ResNet-v1/2 use 5, 10, 5
        """
        super(Simple, self).__init__()
        self.depth = depth

        # down layers
        self.s1 = S1(base, b=bias)  # conv
        self.s2 = S2(base, b=bias)  # 
        self.s3 = S3(base, b=bias)  # 
        self.s4 = S4(base, b=bias)  # 
        self.a_n = nn.ModuleList([A(base, b=bias, lin_act=lin_act) for _ in range(Na)])
        self.ra = redA(base, b=bias)  # 
        self.b_n = nn.ModuleList([B(base, b=bias, lin_act=lin_act) for _ in range(Nb)])
        self.rb = redB(base, b=bias)  # 
        self.c_n = nn.ModuleList([C(base, b=bias, lin_act=lin_act) for _ in range(Nc)])

        # up layers
        self.irb = invRedB(base, b=bias)  #
        self.ib = invB(base, 2, b=bias, attn=ATTN)  #
        self.ira = invRedA(base, b=bias)  #
        self.ia = invA(base, 4, b=bias, attn=ATTN)  #
        self.is4 = invS4(base, b=bias)  #
        self.is3 = invS3(base, 7, b=bias, attn=ATTN)  #
        self.is2 = invS2(base, b=bias)  #
        self.is1 = invS1(base, 13, b=bias, attn=ATTN)  #

        if ROI:  # 53H, 65W
            self.final = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(63, 75)),
                Conv2(base, k=5, s=1, p=0, b=bias, drop_p=drop_p),
                Conv2(base, k=3, s=1, p=0, b=bias, drop_p=drop_p),
                Conv2(base, k=3, s=1, p=0, b=bias, drop_p=drop_p),
                Conv2(base, k=3, s=1, p=0, b=bias, drop_p=drop_p),
                #nn.MaxPool2d(kernel_size=(3, 5), stride=(1, 2)),
                #Conv2(base, k=5, s=1, p=(0, 4), d=3, b=bias, drop_p=drop_p),
                #Conv2(base, k=5, s=1, p=(0, 4), d=2, b=bias, drop_p=drop_p),
                #Conv2(base, k=5, s=1, p=(0, 2), b=bias, drop_p=drop_p),
                #Conv2(base, k=3, s=1, p=0, b=bias, drop_p=drop_p),
                #Conv2(base, k=3, s=1, p=0, b=bias, drop_p=drop_p),
                nn.LazyConv2d(1, kernel_size=1, stride=1, padding=0, bias=bias),
            )
        else:
            self.final = nn.Sequential(
                nn.LazyConv2d(1, kernel_size=1, stride=1, padding=0, bias=bias),
            )

    def forward(self, x):
        # UNet downswing
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)

        a = s4
        for anet in self.a_n:
            a = anet(a)

        ra = self.ra(a)

        b = ra
        for bnet in self.b_n:
            b = bnet(b)

        rb = self.rb(b)

        c = rb
        for cnet in self.c_n:
            c = cnet(c)

        # bottom of the UNet
        # reshape to 2D in/out
        s = c.shape
        c = c.reshape(s[0], -1, s[3], s[4])
        printit(f'BOTTOM: {c.shape}')

        # UNet upswing
        irb = self.irb(c, b.shape[3:])
        ib = self.ib(irb, b)
        ira = self.ira(ib, a.shape[3:])
        ia = self.ia(ira, a)
        is4 = self.is4(ia, s3.shape[3:])
        is3 = self.is3(is4, s3)
        is2 = self.is2(is3, s1.shape[3:])
        is1 = self.is1(is2, s1, x.shape[3:])

        out = self.final(is1)
        printit('FINAL', out.shape)
        return out

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

