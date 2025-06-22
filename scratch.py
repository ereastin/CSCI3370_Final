import xarray as xr
import os
import torch
import sys
sys.path.append('/home/eastinev/AI')
import paths as pth
def main():
    ff = '/projects/bccg/Data/REANA/MERRA2/M2I3NPASM/MERRA2_300.inst3_3d_asm_Np.20020714.nc4'  # MERRA2_200.inst3_3d_asm_Np.19990403.nc4'
    a = '1992084.03.nc' # '2008118.21.nc'
    f = os.path.join(pth.MSWEP, a)
    ds = xr.open_dataset(ff)
    print(ds)

if __name__ == '__main__':
    main()
