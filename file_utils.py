import os
import sys
sys.path.append('/home/eastinev/AI')
import paths as pth
import numpy as np
from functools import partial
from itertools import product
import earthaccess
import boto3
import xarray as xr

## ================================================================================
# Necessary for composing filename requests
DPM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SPLITS = {
    31: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 25), np.arange(25, 32)],
    30: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 24), np.arange(24, 31)],
    29: [np.arange(1, 9), np.arange(9, 16), np.arange(16, 23), np.arange(23, 30)],
    28: [np.arange(1, 8), np.arange(8, 15), np.arange(15, 22), np.arange(22, 29)],
}

CLOUD = False
EXP = 'inc3d_reana'
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

## ================================================================================
def main():
    boto3.setup_default_session(region_name='us-west-2')
    s3_client = boto3.client('s3')
    if (s3_client.meta.region_name != 'us-west-2'):
        print('failed connecting to correct region')

    # Authenticate using Earthdata Login prerequisite files
    auth = earthaccess.login()
    
    t_strs = get_time_strs()

    # get 'temporal'
    for t in t_strs:
        times = make_str(t)

        # Search for granule
        results = earthaccess.search_data(
            short_name='M2I3NVASM',  # model-level data!
            temporal=times,
            # bounding_box=(-110, 24, -70, 52)  # not confident this actually works
        )

        fnames = earthaccess.open(results)

# -----------------------------------------------------------------------------
def make_str(time_id):
    (yr, mn, wk) = time_id
    wk_days = _get_week_days(yr, mn, wk)
    t_start = f'{year}-{str(mn).zfill(2)}-{str(wk_days[0]).zfill(2)}T00'
    t_end = f'{year}-{str(mn).zfill(2)}-{str(wk_days[-1]).zfill(2)}T21'
    return (t_start, t_end)

# -----------------------------------------------------------------------------
def get_time_strs():
    if EXP == 'inc3d_reana':
        n_months = 6 if SEASON == 'both' else 3
        mnth_offset = 6 if SEASON == 'sum' else 3
    else:
        n_months = 8
        mnth_offset = 4

    yrs = list(range(2004, 2021)) #list(range(1980, 2021))
    mnths = list(range(mnth_offset, mnth_offset + n_months))
    wks = list(range(4))

    if WEEKLY:
        t_strs = [(l[0], l[1], l[2]) for l in list(product(yrs, mnths, wks))]
        # remove weeks without a single MCS
        for t in RM_WEEKS:
            if t in t_strs:
                t_strs.pop(t_strs.index(t))
    else:  # monthly
        t_strs = [(l[0], l[1]) for l in list(product(yrs, mnths))]

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
    return merra_fs, mswep_fs

# -----------------------------------------------------------------------------
def _get_filepaths(year, month, week, exp, forecast=False):
    # MERRA stuff
    base = os.path.join('/projects/bccg/Data/ml-for-climate/cus_precip')
    #base = os.path.join(pth.SCRATCH, exp)
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
def write_files(exp):
    curr_files = os.listdir(os.path.join(pth.SCRATCH, exp))
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    merra_files, mswep_files = self._get_files_by_time(year, month, week)
    # write source
    try:
        if not source_rw_pth.split(os.sep)[-1] in curr_files:
            self.write_source(
                source_rw_pth,
                merra_files,
                self.merra_drop_vars,
                self.merra_regrid,
                lat=self.merra_lats,
                lon=self.merra_lons,
                lev=self.p
            )
    except Exception as e:
        print(f'MERRA {source_rw_pth} failed', e)

    # write target
    try:
        if not target_rw_pth.split(os.sep)[-1] in curr_files:
            self.write_target(
                target_rw_pth,
                mswep_files,
                self.mswep_drop_vars,
                self.mswep_regrid,
                lat=self.mswep_lats,
                lon=self.mswep_lons
                # TODO: add lev selection here too for target agnostic work
            )
    except Exception as e:
        print(f'MSWEP {target_rw_pth} failed', e)
    return (None, None, time_id)

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
    ds = self._regrid(ds, regrid_dict)
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
    ds = self._regrid(ds, regrid_dict)
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
