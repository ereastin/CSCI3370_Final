import dask
from dask import array
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import torch
import xarray as xr
import xarray_regrid
import numpy as np
import pandas as pd
import json
import os
import sys
sys.path.append('/home/eastinev/AI')
import utils  # TODO: test the perturbations make sure they work correctly
import paths as pth

KEEP_PIX = [
    'base_time',
    'precipitation',
    'cloudtracknumber',  # mask everywhere assigned to track#, else nan
    'lat',
    'lon',
    'latitude',
    'longitude',
    'time'
]

KEEP_STATS = [
    'base_time',
    'start_basetime',
    'end_basetime',
    'lifecycle_stage',
    'lifecycle_complete_flag',
    'meanlat',
    'meanlon',
    'ccs_area',
    'core_area',
    # 'cloudnumber',
    'total_rain'
]

TOP = 50.5
BOT = 24
RT = -83.125
LFT = -108.75

LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])

# =====================================================================
def main():
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)  # have to specify this think it defaults to 4 or smthn
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        source = xr.open_dataset(os.path.join(pth.SCRATCH, 'cus', 'LSF_2004_04_0.nc'))
        VAR = 'U'
        time_id = {'yr': 2004, 'mn': 4, 'days': torch.arange(1, 9)}
        mask, times = run(time_id)  # this is a DataArray btw
        source = source.sel(time=times)
        perturb_dict = {0: {'var': 'U', 'type': 'shuffle', 'scale': 100, 'region': mask, 'invert': True, 'levels': LEV[:-1]}}
        source = utils.perturb(source, perturb_dict)
        source.to_netcdf('./check.nc', engine='netcdf4')

# ---------------------------------------------------------------------
def get_times(time_id):
    year = time_id['yr']
    good_tracks, track_idx = read_stats(year, time_id)
    # NOTE: track_idx are offset by -1 relative to mask value

    # loop through all tracks..?
    times = np.array([], dtype='datetime64')
    for i in range(len(good_tracks.tracks)):
        track = good_tracks.isel(tracks=i)
        track_stats = get_track_stats(track)
        t = track_stats['base_time'].values[()]
        times = np.append(times, t)

    # TODO: if wanting stats for something need to compile in loop above and pass along
    times = np.unique(times)
    if len(times) == 0:
        return []
    return times

# ---------------------------------------------------------------------
def get_mask(times, time_id):
    year = time_id['yr']
    pix_data = read_pixel_data(year, times)
    mask = _regrid(pix_data['cloudtracknumber'], {'do': True})

    return mask

# ---------------------------------------------------------------------
def run(time_id):
    times = get_times(time_id)
    mask = get_mask(times, time_id)
    return mask, times

# ---------------------------------------------------------------------
def _regrid(ds, regrid_dict):
    if not regrid_dict['do']: return ds
    ds_grid = xr.open_dataset('./vgrid.nc')
    ds = ds.regrid.nearest(ds_grid)
    return ds

# ---------------------------------------------------------------------
def read_pixel_data(year, times):
    """
    Filenames: <dir>/<filename>
        YYYYMMDD.0000_YYYYMMDD.0000/mcstrack_YYYYMMDD_HH0000.nc
        ** THERE MAY BE TIMESTEPS MISSING IF NO MCS IDENTIFIED
        ** For full suite of available MCS statistics see README
    """

    mn1, mn2, yr2 = get_months(year)
    data_dir = f'{year}{mn1}01.0000_{yr2}{mn2}01.0000'
    times = [pd.Timestamp(t) for t in times]
    fnames = [f'mcstrack_{year}{str(t.month).zfill(2)}{str(t.day).zfill(2)}_{str(t.hour).zfill(2)}0000.nc' for t in times]
    data_paths = [os.path.join(pth.MCS_DATA, data_dir, f) for f in fnames]

    with open('pix_lvl_vars.json', 'r') as f:
        pix_vars = json.load(f)['vars']

    for v in KEEP_PIX:
        pix_vars.pop(pix_vars.index(v))

    ds = xr.open_mfdataset(
        data_paths,
        drop_variables=pix_vars,
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
    return ds

# ---------------------------------------------------------------------
def read_stats(year, time_id=None):
    """
    Will return tracks within eval period of entire year if time_id == None
    Otherwise will return tracks within specific week

    Filenames:
        mcs_tracks_final_extc_YYYYMMDD.0000_YYYYMMDD.0000.nc
        -> 2004-2017 all seasons (eg 20170101.0000-20180101.0000)
        -> 2018-2021 warm season (Apr-Sep, eg 20180401.0000-20190901.0000)
    **For full suite of available MCS statistics see README
    """
    mn1, mn2, yr2 = get_months(year)
    path = os.path.join(pth.MCS_STATS, f'mcs_tracks_final_extc_{year}{mn1}01.0000_{yr2}{mn2}01.0000.nc')

    with open('stats_vars.json', 'r') as f:
        stats_vars = json.load(f)['vars']

    for v in KEEP_STATS:
        stats_vars.pop(stats_vars.index(v))

    ds = xr.open_dataset(path, drop_variables=stats_vars)

    # do stats read, select for only MCSs that underwent complete lifecycle and in training period
    complete_idx = ds['lifecycle_complete_flag'] == 1
    if time_id is None:
        time_idx = (ds['start_basetime'] >= np.datetime64(f'{year}-03-01')) & (ds['end_basetime'] <= np.datetime64(f'{year}-08-30'))
    else:
        mn = str(time_id['mn']).zfill(2)
        d0, d1 = str(time_id['days'][0].item()).zfill(2), str(time_id['days'][-1].item()).zfill(2)
        time_idx = (ds['start_basetime'] >= np.datetime64(f'{year}-{mn}-{d0}T00:00')) & (ds['end_basetime'] <= np.datetime64(f'{year}-{mn}-{d1}T21:00'))

    select = complete_idx & time_idx
    good_tracks = ds.isel(tracks=select)
    track_idx = [i for i, v in enumerate(select) if v == True]  # need these to identify MCS mask in pixel level data

    return good_tracks, track_idx

# ---------------------------------------------------------------------
def get_track_stats(track):
    # hellish way to get timestamps in MSWEP dataset
    base_time = track['base_time']
    mask = np.isnat(base_time.values[()])
    track = track.isel(times=~mask)  # select data-containing timesteps for all tracks

    eval_time = np.array([t if pd.Timestamp(t).hour in list(range(0, 22, 3)) else np.datetime64('nat') for t in track['base_time'].values[()]])
    time_mask = eval_time == track['base_time'].values[()]

    '''
    # before sampling only MSWEP times agg total rain
    # also have convective and stratiform rain if that is useful
    total_rain = track['total_rain'].values[()] # need to agg from previous timesteps
    for i, val in enumerate(eval_mask):
        if val:
            agg = total_rain[i-2:i+1]
            track['total_rain'].loc[dict(times=i)] = np.sum(agg)
    '''

    #lat, lon = track['meanlat'].values[()], track['meanlon'].values[()] 
    #good_lat = np.where((lat <= TOP) & (lat >= BOT), True, False)
    #good_lon = np.where((lon <= RT) & (lon >= LFT), True, False)
    # sample only MSWEP times AND in model region 'view'
    #region_mask = good_lat & good_lon
    #eval_mask = region_mask & time_mask
    track = track.isel(times=time_mask)

    return track

# ---------------------------------------------------------------------
def get_months(year):
    if year >= 2018:
        mn1, mn2, yr2 = 4, 9, year
    else:
        mn1, mn2, yr2 = 1, 1, year + 1
    mn1, mn2 = str(mn1).zfill(2), str(mn2).zfill(2)
    return mn1, mn2, yr2

# =====================================================================
if __name__ == '__main__':
    main()
