import xarray as xr
import xarray_regrid
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from xgcm import Grid
import numpy as np
import os
import sys
from functools import partial
sys.path.append('/home/eastinev/AI')
import paths as pth
import utils

## ================================================================================
DROP_VARS = ['TS', 'TREFHT', 'QREFHT', 'PS', 'PRECC', 'PRECL']
# CESM2 data
# CTRL: fhist_sglc.CTRL.cam.h0.YYYY-MM.3hrly.mergetime.nc
# 4K: fhist_sglc.2K.cam.h0.YYYY-MM.3hrly.mergetime.nc
# 'lev': 3.643466,   7.59482,  14.356632,  24.61222,  35.92325,  43.19375, 51.677499,  61.520498,  73.750958,  87.82123 , 103.317127, 121.547241, 142.994039, 168.22508 , 197.908087, 232.828619, 273.910817, 322.241902, 379.100904, 445.992574, 524.687175, 609.778695, 691.38943 , 763.404481, 820.858369, 859.534767, 887.020249, 912.644547, 936.198398, 957.48548, 976.325407, 992.556095])
# 'lev_bnds': <xarray.Variable (lev: 32, bnds: 2)> something... 

LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])

CONUS_LON = slice(225 - 1, 315 + 1)
CONUS_LAT = slice(18 - 1, 58 + 1)
CUS_LON = slice(-110 - 1, -70 + 1)
CUS_LAT = slice(25 - 1, 51 + 1)

SEL_VARS = ['Z3', 'Q', 'T', 'U', 'V', 'OMEGA']
## ================================================================================
def main():
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus)
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)

        files_4x = os.listdir(pth.CESM4X)
        #preproc_fn = partial(_select_batch, lat=CONUS_LAT, lon=CONUS_LON)
        ds_4x = xr.open_dataset(
            os.path.join(pth.CESM4X, files_4x[0]),
            #preprocess=preproc_fn,
            #drop_variables=DROP_VARS,
            #parallel=True,
            #chunks='auto',
            #engine='h5netcdf'
        )
        ds_4x = ds_4x.sel(lat=CONUS_LAT, lon=CONUS_LON)

        # extract/convert precip, regrid conservative -> target
        # this works and is relatively quick... LSF not so much
        '''
        target_ds = convert_precip(ds_4x.PRECT)
        target_ds = utils.regrid(
            target_ds,
            {'do': True,
            'target_grid': '~/AI/incept/pgrid.nc',
            'regrid_type': 'conservative'}
        )
        print(target_ds)
        '''
        source_da = ds_4x[SEL_VARS]
        print(source_da)
        exit()
         
        grid = Grid(
            source_da,
            coords={
                'X': {'center': 'lon'},
                'Y': {'center': 'lat'},
                'Z': {'center': 'lev'}
            },
            periodic=False,
            autparse_metadata=False
        )
        T = grid.transform(
            source_da['T'],
            'Z',
            LEV,
            target_data=source_da['P']
        )
        source_da = utils.regrid(
            source_da,
            {'do': True,
            'target_grid': '~/AI/incept/vlgrid.nc',
            'regrid_type': 'linear'}
        )
        print(source_da)

    return
    files_ctrl = os.listdir(pth.CESMCTRL)
    ds_ctrl = xr.open_dataset(
        os.path.join(pth.CESMCTRL, files_ctrl[0]),
        drop_variables=DROP_VARS,
    )

def _select_batch(ds, **kwargs):
    return ds.sel(**kwargs)

def convert_precip(ds): # TODO: better to use total or convective, stratiform.?
    # TODO: how can this be turned into surface precip.?
    # this in m/s -> mm/3hr
    ds *= (3600 * 1000 * 3)
    return ds

## ================================================================================
if __name__ == '__main__':
    main()

