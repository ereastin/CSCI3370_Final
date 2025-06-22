import torch
import torch.nn as nn

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.net = nn.LazyConv3d(1, kernel_size=1)

    def forward(self, x):
        x = self.net(x)
        out = x[:, 0, 0, :54, :42].unsqueeze(1)
        return out

