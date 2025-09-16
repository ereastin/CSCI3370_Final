import os
import sys
import numpy as np
import pandas as pd
from functools import partial
from itertools import product
import earthaccess
import boto3
import xarray as xr
import xarray_regrid
import dask
from dask import array
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from cdo import *
import time
import subprocess

sys.path.append('/home/eastinev/AI')
import paths as pth
import utils

## ================================================================================
# Necessary for composing filename requests
DPM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DT_SPLITS = {
    31: [
        np.arange(0, 8, dtype='timedelta64[D]'),
        np.arange(8, 16, dtype='timedelta64[D]'),
        np.arange(16, 24, dtype='timedelta64[D]'),
        np.arange(24, 31, dtype='timedelta64[D]')
    ],
    30: [
        np.arange(0, 8, dtype='timedelta64[D]'),
        np.arange(8, 16, dtype='timedelta64[D]'),
        np.arange(16, 23, dtype='timedelta64[D]'),
        np.arange(23, 30, dtype='timedelta64[D]')
    ],
    29: [
        np.arange(0, 8, dtype='timedelta64[D]'),
        np.arange(8, 15, dtype='timedelta64[D]'),
        np.arange(15, 22, dtype='timedelta64[D]'),
        np.arange(22, 29, dtype='timedelta64[D]')
    ],
    28: [
        np.arange(0, 7, dtype='timedelta64[D]'),
        np.arange(7, 14, dtype='timedelta64[D]'),
        np.arange(14, 21, dtype='timedelta64[D]'),
        np.arange(21, 28, dtype='timedelta64[D]')
    ], 
}

#YEARS = list(range(2004, 2021))
YEARS = list(range(1979, 1985))
MONTHS = [9] # list(range(3, 9))
WEEKS = [0] # list(range(4))

CLOUD = False
EXP = 'cus'
SEASON = 'spr'

DROP_VARS = ['EPV', 'SLP', 'PS', 'RH', 'PHIS', 'O3', 'QI', 'QL']
# MCS DB is on lat(25, 51), lon(-110, -70)
CONUS_LON = slice(-135, -45) # (-135, -45.625)
CONUS_LAT = slice(18, 58) # (18.5, 58)
CUS_LON = slice(-110 - 1, -70 + 1)
CUS_LAT = slice(51 + 1, 25 - 1)  # mswep has these backwards, need buffer for remap
F = 'CTRL'

'''
# CUS
# MERRA selection criteria
self.merra_regrid = {'do': False, 'target_grid': None}
self.merra_vars = ['U', 'V', 'OMEGA', 'H', 'T', 'QV']  # if interested could use cloud ice and liquid mass mixing ratios?

# MSWEP selection criteria
self.mswep_regrid = {'do': True, 'target_grid': './pgrid.nc'} 
'''

LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])
## ================================================================================
def main():
    #args = sys.argv
    #F = args[1]

    t1 = time.time()
    cdo = Cdo(tempdir=pth.TMP)
    cdo.debug = True
    t_strs = get_time_strs()

    fs = os.listdir(os.path.join(pth.SCRATCH, 'cus_cesm'))
    cesm_t_strs = []
    for f in fs:
        if F in f and 'LSF' in f:
            a = f.split('_')[1:4]
            yr, mn, wk = int(a[0]), int(a[1]), int(a[2][0])
            cesm_t_strs.append((yr, mn, wk))
    # these should be the missing ones
    #for t in t_strs:
    #    if t not in cesm_t_strs:
    #        print(t)
    #exit()

    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)

        # do for CESM data
        for t in t_strs:
            if t in cesm_t_strs:
                continue
            yr, mn, wk = t[0], t[1], t[2]
            p_out = os.path.join(pth.SCRATCH, 'cus_cesm', f'P_{yr}_{str(mn).zfill(2)}_{wk}.{F}.nc')
            lsf_out = os.path.join(pth.SCRATCH, 'cus_cesm', f'LSF_{yr}_{str(mn).zfill(2)}_{wk}.{F}.nc')
            curr_files = os.listdir(os.path.join(pth.SCRATCH, 'cus_cesm'))
            if p_out.split(os.sep)[-1] in curr_files:
                print(f'skipping {t}', flush=True)
                continue
            fs = _get_cesm_by_time(yr, mn, wk, exp=F)
            #fs = _get_cesm_merge_by_time(yr, mn, wk, exp=F)
            dt_range = _get_wk_days_no_leap(yr, mn, wk)
            print(fs, dt_range)
            a, b = dt_range[0].astype(str), dt_range[-1].astype(str)
            try:
                for f in fs:
                    tmp_out = os.path.join(pth.TMP, f'tmp.{np.random.randint(10000, 99999)}.nc')
                    ds = xr.open_dataset(f)
                    #t_slc = slice(a, b)
                    #lat_slc = slice(17, 59)
                    #lon_slc = slice(360-136, 360-44)
                    #ds = ds.sel(time=t_slc, lat=lat_slc, lon=lon_slc)
                    if not hasattr(ds['lev'], 'bounds') and not hasattr(ds['lev'], 'formula'):
                        print(f'skipping {f}', flush=True)
                        ds['lev'].attrs['bounds'] = "ilev"
                        ds['lev'].attrs['formula'] = "a*p0+b*ps"

                        #v_sel = ['U', 'V', 'OMEGA', 'Z3', 'T', 'Q', 'PRECT', 'P0', 'PS', 'hyam', 'hybm', 'hyai', 'hybi', 'time_bnds']
                        #ds = ds[v_sel]
                        #ds.to_netcdf(tmp_out)
                        ds.to_netcdf(f, mode='a')
                        ds.close()

                print('performing level interpolation...', flush=True)
                tmp_out2 = os.path.join(pth.TMP, f'tmp2.{np.random.randint(10000, 99999)}.nc')
                cdo.ml2pl(
                    ','.join((LEV * 100).astype(str)),
                    input=f'-mergetime [ -selname,U,V,OMEGA,Z3,T,Q,PRECT -sellonlatbox,-136,-44,17,59 : {" ".join(fs)} ]',  # for hourly data
                    #input=f'{tmp_out}',
                    output=tmp_out2,
                    options='-f nc4'
                )
                ds = xr.open_dataset(tmp_out2, chunks={}, engine='h5netcdf')
                ds['plev'] = ds['plev'] / 100
                ds['plev'].attrs['units'] = 'hPa'

                # do xarray stuff
                print('writing large scale...', flush=True)
                write_cesm_lsf(
                    ds,
                    lsf_out,
                    {
                        'do': True,
                        'target_grid': '~/AI/incept/vgrid.nc',
                        'regrid_type': 'linear'  # is this bilinear.?
                    },
                )
                print('writing precip...', flush=True)
                write_cesm_precip(
                    ds,
                    p_out,
                    {
                        'do': True,
                        'target_grid': '~/AI/incept/pgrid.nc',
                        'regrid_type': 'conservative'
                    },
                )
            except FileNotFoundError:
                print(f'FileNotFound, skipping {t}', flush=True)
                continue
            except KeyboardInterrupt:
                print('exiting', flush=True)
                break
            clean_temp()

        clean_temp()
        t2 = time.time()
        print(t2 - t1, flush=True)
        cdo.cleanTempDir()
        return

        # do for MSWEP/MERRA
        for t in t_strs:
            times = make_str(t)
            yr, mn, wk = t[0], t[1], t[2]
            curr_files = os.listdir(os.path.join(pth.SCRATCH, EXP))
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
            source_write_path, target_write_path = _get_filepaths(yr, mn, wk, EXP)
            #if source_write_path.split(os.sep)[-1] in curr_files:
            #    continue
            #else:
            print(f'writing {target_write_path}')

            source_files, target_files = _get_files_by_time(yr, mn, wk)

            '''
            write_source(
                source_write_path,
                source_files,
                DROP_VARS,
                {'do': False},
                lat=CONUS_LAT,
                lon=CONUS_LON,
                lev=LEV,
            )
            '''
            write_target(
                target_write_path[0],
                target_files,
                [],
                {
                    'do': True,
                    'target_grid': '~/AI/incept/pgrid.nc',
                    'regrid_type': 'conservative'
                },
                lat=CUS_LAT,
                lon=CUS_LON
            )

# -----------------------------------------------------------------------------
def clean_temp():
    fs = os.listdir(pth.TMP)
    for f in fs:
        if 'tmp' in f:
            os.remove(os.path.join(pth.TMP, f))

# -----------------------------------------------------------------------------
def make_str(time_id):
    (yr, mn, wk) = time_id
    wk_days, _ = _get_wk_days(yr, mn, wk)
    t_start = f'{yr}-{str(mn).zfill(2)}-{str(wk_days[0]).zfill(2)}T00'
    t_end = f'{yr}-{str(mn).zfill(2)}-{str(wk_days[-1]).zfill(2)}T21'
    return (t_start, t_end)

# -----------------------------------------------------------------------------
def get_time_strs():
    t_strs = [(l[0], l[1], l[2]) for l in list(product(YEARS, MONTHS, WEEKS))]
    return t_strs

# -----------------------------------------------------------------------------
def _get_wk_days_leap(year, month, week):
    n_days = DPM[month - 1]
    n_days += 1 if (year % 4 == 0) and month == 2 else 0
    dates = DT_SPLITS[n_days][week] + np.datetime64(f'{year}-{str(month).zfill(2)}')
    return dates

# -----------------------------------------------------------------------------
def _get_wk_days_no_leap(year, month, week):
    # CESM data uses no leap years AFAIK
    n_days = DPM[month - 1]
    dates = DT_SPLITS[n_days][week] + np.datetime64(f'{year}-{str(month).zfill(2)}')
    return dates

# -----------------------------------------------------------------------------
def _get_merra_nsel(year, month):
    # N is 100, 200, 300, 400, 401, for 1980-1991, 2 for 1992-2000, 3 for 2001-2010, 4 for 2011+ 
    if year in list(range(1980, 1992)):
        N = 100
    elif year in list(range(1992, 2001)):
        N = 200
    elif year in list(range(2001, 2011)):
        N = 300
    elif year == 2020 and month == 9:
        N = 401
    else:
        N = 400
    return N

# -----------------------------------------------------------------------------
def _get_files_by_time(year, month, week):
    dt_range = _get_wk_days_leap(year, month, week)
    N = _get_merra_nsel(year, month)
    merra_fs = _get_merra_by_time(dt_range, N)
    mswep_fs = _get_mswep_by_time(dt_range)
    return merra_fs, mswep_fs

# -----------------------------------------------------------------------------
def _get_merra_by_time(dt_range, N):
    # MERRA fname: <>.YYYYMMDD.nc4
    dt_strs = [pd.Timestamp(dt).strftime('%Y%m%d') + '.nc4' for dt in dt_range]
    merra_base = f'MERRA2_{N}.inst3_3d_asm_Np.'
    merra_fs = [os.path.join(pth.MERRA,  merra_base + dt) for dt in dt_strs]
    return merra_fs

# -----------------------------------------------------------------------------
def _get_mswep_by_time(dt_range, forecast_step=0):
    dt_range = np.arange(dt_range[0], dt_range[-1] + 1, np.timedelta64(3, 'h'))
    dt_range += np.timedelta64(3 * forecast_step, 'h')
    # MSWEP fname: YYYYDDD.HH.nc
    dt_range = [pd.Timestamp(t).strftime('%Y%j.%H.nc') for t in dt_range]
    mswep_fs = [os.path.join(pth.MSWEP, dt) for dt in dt_range]
    return mswep_fs

# -----------------------------------------------------------------------------
def _get_cesm_by_time(year, month, week, exp='CTRL'):
    dt_range = _get_wk_days_no_leap(year, month, week)
    pwd = pth.CESM4X if exp == '2K' else pth.CESMCTRL
    cesm_fs = [os.path.join(pwd, f'fhist_sglc.{exp}.cam.h0.{dt.astype(str)}-00000.select.nc') for dt in dt_range]
    return cesm_fs

# -----------------------------------------------------------------------------
def _get_cesm_merge_by_time(year, month, week, exp='CTRL'):
    dt_range = _get_wk_days_no_leap(year, month, week)
    pwd = pth.CESM4X if exp == '2K' else pth.CESMCTRL
    cesm_fs = os.path.join(pwd, f'fhist_sglc.{exp}.cam.h0.{year}-{str(month).zfill(2)}.select.mergetime.nc')
    return [cesm_fs]

# -----------------------------------------------------------------------------
def _get_filepaths(year, month, week, exp, forecast=False, cesm_exp=''):
    # MERRA stuff
    base = os.path.join(pth.SCRATCH, exp)
    source_rw_pth = os.path.join(base, f'LSF_{year}_{str(month).zfill(2)}_{week}{cesm_exp}.nc')

    # MSWEP stuff
    if not forecast:
        target_rw_pth = [os.path.join(base, f'P_{year}_{str(month).zfill(2)}_{week}{cesm_exp}.nc')]
    else:
        if week == 3:
            paths = [f'P_{year}_{str(month).zfill(2)}_{week}{cesm_exp}.nc', f'P_{year}_{str(month + 1).zfill(2)}_{0}{cesm_exp}.nc']
        else:
            paths = [f'P_{year}_{str(month).zfill(2)}_{week}{cesm_exp}.nc', f'P_{year}_{str(month).zfill(2)}_{week + 1}{cesm_exp}.nc']
        target_rw_pth = [os.path.join(base, p) for p in paths]

    return source_rw_pth, target_rw_pth

# -----------------------------------------------------------------------------
def _select_batch(ds, **kwargs):
    return ds.sel(**kwargs)

# -----------------------------------------------------------------------------
def write_target(out_file, target_files, drop_vars, regrid_dict, **kwargs):
    preproc_fn = partial(_select_batch, **kwargs)
    ds = xr.open_mfdataset(
        target_files,
        preprocess=preproc_fn,
        drop_variables=drop_vars,
        concat_dim='time',
        data_vars='minimal',
        coords='minimal',
        combine='nested',
        compat='override',
        join='override',
        parallel=True,
        chunks='auto',
        engine='h5netcdf'
    )
    ds = utils.regrid(ds, regrid_dict)
    ds = ds.compute()
    ds.to_netcdf(out_file, engine='netcdf4')

# -----------------------------------------------------------------------------
def read_target(target_files, drop_vars, regrid_dict, **kwargs):
    preproc_fn = partial(_select_batch, **kwargs)
    ds = xr.open_mfdataset(
        target_files,
        preprocess=preproc_fn,
        drop_variables=drop_vars,
        concat_dim='time',
        data_vars='minimal',
        coords='minimal',
        combine='nested',
        compat='override',
        join='override',
        parallel=True,
        chunks='auto',
        engine='h5netcdf'
    )
    ds = utils.regrid(ds, regrid_dict)
    return ds

# -----------------------------------------------------------------------------
def write_source(out_file, source_files, drop_vars, regrid_dict, **kwargs):
    preproc_fn = partial(_select_batch, **kwargs)
    ds = xr.open_mfdataset(
        source_files,
        preprocess=preproc_fn,
        drop_variables=drop_vars,
        concat_dim='time',
        data_vars='minimal',
        coords='minimal',
        combine='nested',
        compat='override',
        join='override',
        parallel=True,
        chunks='auto',
        engine='h5netcdf'
    )
    # do nan interp
    ds0 = ds[['U', 'V', 'OMEGA', 'QV']].fillna(value=0)
    # NOTE: not sure this is the best way to handle but other options? interpolate below ground?
    dsBF = ds[['T', 'H']].bfill(dim='lev')
    ds = xr.merge([ds0, dsBF])
    ds = utils.regrid(ds, regrid_dict)
    ds = ds.compute()
    ds.to_netcdf(out_file, engine='netcdf4')

# -----------------------------------------------------------------------------
def write_cesm_lsf(source_ds, out_f, regrid_dict):
    # select for 3 hourly data
    source_ds = source_ds.isel(time=slice(None, None, 3))
    # do nan interp
    ds0 = source_ds[['U', 'V', 'OMEGA', 'Q']].fillna(value=0)
    # NOTE: not sure this is the best way to handle but other options? interpolate below ground?
    # how do other ML papers handle this? all just on model levels?
    # can make argument that terrain is 'built in'?
    dsBF = source_ds[['T', 'Z3']].bfill(dim='plev')
    ds = xr.merge([ds0, dsBF])
    # think this is the correct order instead of regridding on model levels
    ds = utils.regrid(ds, regrid_dict)
    ds = ds.compute()
    ds = ds.astype(np.float32)
    ds.to_netcdf(out_f, engine='netcdf4')

# -----------------------------------------------------------------------------
def write_cesm_precip(source_ds, out_f, regrid_dict):
    source_ds['PRECT'].attrs['units'] = 'mm'
    precip = source_ds['PRECT']  # need to resample to calc accum for each 1 hour step and then sum for 3 hourly
    precip *= (1000 * 3600)  # in m/s --> mm/hr
    precip = precip.resample(time='3h').sum()
    precip = utils.regrid(precip, regrid_dict)
    precip = precip.compute()
    precip = precip.astype(np.float32)
    precip.to_netcdf(out_f, engine='netcdf4')

# -----------------------------------------------------------------------------
def cloud_data():
    # unused atm moment, wont work for ML stuff its too slow
    # need to look at things in the TRACMIP/Pangeo intake stuff?
    boto3.setup_default_session(region_name='us-west-2')
    s3_client = boto3.client('s3')
    if (s3_client.meta.region_name != 'us-west-2'):
        print('failed connecting to correct region')

    # Authenticate using Earthdata Login prerequisite files
    auth = earthaccess.login()
    # Search for granule
    results = earthaccess.search_data(
        short_name='M2I3NVASM',  # model-level data!
        temporal=times,
        # bounding_box=(-110, 24, -70, 52)  # not confident this actually works
    )
    fnames = earthaccess.open(results)

## ================================================================================
if __name__ == '__main__':
    main()
