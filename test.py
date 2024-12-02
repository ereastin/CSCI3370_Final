import xarray as xr
import os
from PrecipDataset import PrecipDataset
import torch
from torch.utils.data import DataLoader

def main():
    pd = PrecipDataset('train')
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loader = DataLoader(pd, batch_size=None, pin_memory=True) 
    for i, (x, y, t) in enumerate(loader):
        if i == 5:
            break
        print(t)
        x = x.to(dev)
        y = y.to(dev)
    # do some looking to make sure scratch dir stuff is right
    # verify interleaved source data
    # verify dates aligned appropriately
    # verify cdo actually works..

if __name__ == '__main__':
    main()

