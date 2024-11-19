import xarray as xr
import os
from PrecipDataset import PrecipDataset

def main():
    pd = PrecipDataset('train')
    for i in range(len(pd)):
        _ = pd.__getitem__(i)
    # do some looking to make sure scratch dir stuff is right
    # verify interleaved source data
    # verify dates aligned appropriately
    # verify cdo actually works..

if __name__ == '__main__':
    main()

