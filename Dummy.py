import torch

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy).__init__()

    def forward(self, x):
        return x

