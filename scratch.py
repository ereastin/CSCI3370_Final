import xarray as xr
import os

def main():
    f = '/scratch/eastinev/inc3d_reana/P.200003_3.nc'
    ds = xr.open_dataset(f)
    print(ds['lon'])

if __name__ == '__main__':
    main()
