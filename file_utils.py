import os
import sys
import numpy as np
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

sys.path.append('/home/eastinev/AI')
import paths as pth
import utils

## ================================================================================
# Necessary for composing filename requests
DPM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SPLITS = {
    31: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 25), np.arange(25, 32)],
    30: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 24), np.arange(24, 31)],
    29: [np.arange(1, 9), np.arange(9, 16), np.arange(16, 23), np.arange(23, 30)],
    28: [np.arange(1, 8), np.arange(8, 15), np.arange(15, 22), np.arange(22, 29)],
}
YEARS = list(range(2004, 2021))
MONTHS = list(range(3, 9))
WEEKS = list(range(4))

CLOUD = False
EXP = 'cus'
SEASON = 'spr'
WEEKLY = True
# weeks without a single detected MCS
RM_WEEKS = [
    (2010, 3, 0), (2019, 3, 2), (2018, 3, 3), (2018, 3, 1), (2020, 3, 2),
    (2015, 3, 2), (2019, 3, 3), (2015, 3, 0), (2018, 4, 2), (2018, 3, 0),
    (2020, 3, 3), (2019, 3, 0), (2012, 3, 0), (2014, 3, 3), (2018, 4, 3),
    (2018, 3, 2), (2006, 3, 0), (2014, 3, 1), (2020, 3, 0), (2013, 3, 0),
    (2019, 3, 1), (2020, 3, 1), (2004, 3, 1), (2007, 4, 1)
]

DROP_VARS = ['EPV', 'SLP', 'PS', 'RH', 'PHIS', 'O3', 'QI', 'QL']
# DROP_VARS = ['CLOUD', 'EPV', 'DELP', 'O3', 'RH', 'SLP', 'PHIS', 'PS', 'QI', 'QL']  # 'PL' mid-level pressure, these for model level
# MCS DB is on lat(25, 51), lon(-110, -70)
CONUS_LON = slice(-135, -45) # (-135, -45.625)
CONUS_LAT = slice(18, 58) # (18.5, 58)
CUS_LON = slice(-110 - 1, -70 + 1)
CUS_LAT = slice(51 + 1, 25 - 1)  # mswep has these backwards, need buffer for remap
# for model level: 150, 208, 288, 375, 450, 562, 637, 700, 750, 800, 820, 850, 880, 910, 955, 985
# LEV = np.array([41, 43, 45, 47, 49, 52, 54, 56, 58, 60, 61, 63, 65, 67, 70, 72])

'''
# CUS
# MERRA selection criteria
self.merra_regrid = {'do': False, 'target_grid': None}
#self.merra_lons = slice(-135, -45)
#self.merra_lats = slice(18, 58)
self.merra_lats = slice(14, 53.5)
self.merra_lons = slice(-140, -50.625)
self.merra_drop_vars = ['EPV', 'SLP', 'PS', 'RH', 'PHIS', 'O3', 'QI', 'QL'] # NOTE: can use RH instead of Q..
self.merra_vars = ['U', 'V', 'OMEGA', 'H', 'T', 'QV']  # if interested could use cloud ice and liquid mass mixing ratios?

# MSWEP selection criteria
self.mswep_regrid = {'do': True, 'target_grid': './pgrid.nc'}  # TODO: fix this to match 
buff = 1
#self.mswep_lats = slice(51 + buff, 25 - buff)
#self.mswep_lons = slice(-110 - buff, -70 + buff)
#self.mswep_regrid = {'do': True, 'extent': (25, 51, -110, -70), 'steps': (0.5, 0.625)}
self.mswep_lats = slice(50.5 + buff, 24 - buff)
self.mswep_lons = slice(-108.75 - buff, -83.125 + buff)
self.mswep_drop_vars = []

# sumatra squall
#self.merra_lats = slice(-15.5, 24)
#self.merra_lons = slice(55, 144.625)
#self.mswep_regrid = {'do': False, 'extent': None, 'steps': None}
#self.mswep_lats = slice(8.0, 0.0)
#self.mswep_lons = slice(92.8, 107.2)
'''

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
#
#self.p = np.array([
#    1000, 975, 950, 925, 900, 875, 850,
#    800, 750, 650, 550, 450, 350, 250, 150, 100
#])
## ================================================================================
def main():
    if CLOUD:
        boto3.setup_default_session(region_name='us-west-2')
        s3_client = boto3.client('s3')
        if (s3_client.meta.region_name != 'us-west-2'):
            print('failed connecting to correct region')

        # Authenticate using Earthdata Login prerequisite files
        auth = earthaccess.login()
    
    t_strs = get_time_strs()

    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus)
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)

        # get 'temporal'
        for t in t_strs:
            times = make_str(t)
            yr, mn, wk = t[0], t[1], t[2]
            curr_files = os.listdir(os.path.join(pth.SCRATCH, EXP))
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
            source_write_path, target_write_path = _get_filepaths(yr, mn, wk, EXP)
            if source_write_path.split(os.sep)[-1] in curr_files:
                continue
            else:
                print(f'writing {source_write_path}')

            source_files, target_files = _get_files_by_time(yr, mn, wk)

            '''
            ds = xr.open_mfdataset(source_files)
            ds = ds.sel(lat=CONUS_LAT, lon=CONUS_LON)
            ds = ds.isel(lev=0, drop=True)
            t = ds.coords['time'].values[()][0]
            da = ds['T']
            da = da.sel(time=[t])
            da.to_netcdf('./vgrid.nc', engine='netcdf4')
            exit()
            '''

            if CLOUD:
                # Search for granule
                results = earthaccess.search_data(
                    short_name='M2I3NVASM',  # model-level data!
                    temporal=times,
                    # bounding_box=(-110, 24, -70, 52)  # not confident this actually works
                )
                fnames = earthaccess.open(results)

            write_source(
                source_write_path,
                source_files,
                DROP_VARS,
                {'do': False},
                lat=CONUS_LAT,
                lon=CONUS_LON,
                lev=LEV,
            )
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
def make_str(time_id):
    (yr, mn, wk) = time_id
    wk_days, _ = _get_wk_days(yr, mn, wk)
    t_start = f'{yr}-{str(mn).zfill(2)}-{str(wk_days[0]).zfill(2)}T00'
    t_end = f'{yr}-{str(mn).zfill(2)}-{str(wk_days[-1]).zfill(2)}T21'
    return (t_start, t_end)

# -----------------------------------------------------------------------------
def get_time_strs():
    if WEEKLY:
        t_strs = [(l[0], l[1], l[2]) for l in list(product(YEARS, MONTHS, WEEKS))]
        # remove weeks without a single MCS
        #for t in RM_WEEKS:
        #    if t in t_strs:
        #        t_strs.pop(t_strs.index(t))
    else:  # monthly
        t_strs = [(l[0], l[1]) for l in list(product(YEARS, MONTHS))]

    return t_strs

# -----------------------------------------------------------------------------
def _get_wk_days(year, month, week=None):
    leap_year = (year % 4 == 0)
    n_days = DPM[month - 1]
    n_days += 1 if leap_year and month == 2 else 0
    wk_days = n_days if week == None else SPLITS[n_days][week]
    return wk_days, leap_year

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
    wk_days, leap_year = _get_wk_days(year, month, week)

    # MERRA fname: <>.YYYYMMDD.nc4
    N = _get_merra_nsel(year, month)
    merra_base = f'MERRA2_{N}.inst3_3d_asm_Np.' + str(year) + str(month).zfill(2)
    merra_fs = [os.path.join(pth.MERRA,  merra_base + str(day).zfill(2) + '.nc4') for day in wk_days]

    # MSWEP fname: YYYYDDD.HH.nc
    d0 = 1 if leap_year and month > 2 else 0
    day_sum = sum(DPM[:month - 1])
    doys = [os.path.join(pth.MSWEP, str(year) + str(day_sum + day).zfill(3)) for day in wk_days + d0]
    hrs = ['.' + str(hr).zfill(2) + '.nc' for hr in range(0, 24, 3)]
    mswep_fs = [f[0] + f[1] for f in list(product(doys, hrs))]
    # remove first time and add another at end
    if wk_days[-1] in [28, 29, 30, 31]:
        day_sum = sum(DPM[:month])
        day = 1 + d0
    else:
        day = wk_days[-1] + 1 + d0
    add_day = os.path.join(pth.MSWEP, str(year) + str(day_sum + day).zfill(3) + '.00.nc')
    mswep_fs.pop(0)
    mswep_fs.append(add_day)
    return merra_fs, mswep_fs

# -----------------------------------------------------------------------------
def _get_filepaths(year, month, week, exp, forecast=False):
    # MERRA stuff
    #base = os.path.join('/projects/bccg/Data/ml-for-climate/cus_precip')
    base = os.path.join(pth.SCRATCH, exp)
    source_rw_pth = os.path.join(base, f'LSF_{year}_{str(month).zfill(2)}_{week}.nc')

    # MSWEP stuff
    if not forecast:
        target_rw_pth = [os.path.join(base, f'P_{year}_{str(month).zfill(2)}_{week}.nc')]
    else:
        if week == 3:
            paths = [f'P_{year}_{str(month).zfill(2)}_{week}.nc', f'P_{year}_{str(month + 1).zfill(2)}_{0}.nc']
        else:
            paths = [f'P_{year}_{str(month).zfill(2)}_{week}.nc', f'P_{year}_{str(month).zfill(2)}_{week + 1}.nc']
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
def write_source_cloud(out_file, **kwargs):
    preproc_fn = partial(_select_batch, **kwargs)
    ds = xr.open_mfdataset(
        source_files,
        preprocess=preproc_fn,
        drop_variables=['CLOUD', 'EPV', 'DELP', 'O3', 'RH', 'SLP', 'PHIS', 'PL', 'PS', 'QI', 'QL'],
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
    ds = ds.compute()
    ds.to_netcdf(out_file, engine='netcdf4')

## ================================================================================
if __name__ == '__main__':
    main()
