import xarray as xr
import os
import torch
def main():
    f = '/scratch/eastinev/inc3d_reana/P.200003.nc'
    ds = xr.open_dataset(f)
    out = torch.tensor(ds.to_dataarray().data)
    print(out.shape)

if __name__ == '__main__':
    main()
