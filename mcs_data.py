import dask
from dask import array
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import xarray as xr
import xarray_regrid
import numpy as np
import pandas as pd
import json
import os
import sys
sys.path.append('/home/eastinev/AI')

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
# =====================================================================
def main():
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)  # have to specify this think it defaults to 4 or smthn
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        year = 2004
        time_id = {'mn': 4, 'wk_days': np.arange(1, 9)}
        run(year, time_id)

# ---------------------------------------------------------------------
def run(time_id):
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
    pix_data = read_pixel_data(year, times)
    # track_nums = np.unique(pix_data['cloudtracknumber'].values[()])  # make sure to exclude nan
    mask = _regrid(pix_data['cloudtracknumber'], {'do': True})
    # select specific masks? use all?
    # mask = pix_data['cloudtracknumber'] == track_idx[i] + 1
    return mask

# ---------------------------------------------------------------------
def _regrid(ds, regrid_dict):
    if not regrid_dict['do']: return ds
    ds_grid = xr.open_dataset('./pgrid.nc')
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

    mn1, mn2 = get_months(year)
    data_dir = f'{year}{mn1}01.0000_{year+1}{mn2}01.0000'
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
    mn1, mn2 = get_months(year)
    path = os.path.join(pth.MCS_STATS, f'mcs_tracks_final_extc_{year}{mn1}01.0000_{year+1}{mn2}01.0000.nc')

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
        d0, d1 = str(time_id['wk_days'][0]).zfill(2), str(time_id['wk_days'][-1]).zfill(2)
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
    eval_mask = eval_time == track['base_time'].values[()]

    # before sampling only MSWEP times agg total rain
    # also have convective and stratiform rain if that is useful
    total_rain = track['total_rain'].values[()] # need to agg from previous timesteps
    for i, val in enumerate(eval_mask):
        if val:
            agg = total_rain[i-2:i+1]
            track['total_rain'].loc[dict(times=i)] = np.sum(agg)

    # sample only MSWEP times
    track = track.isel(times=eval_mask)

    # any other interesting /valuable ones?
    #stages = track['lifecycle_stage'].values[()]
    # cn = track['cloudnumber'].values[()]  # i guess this changes across lifecycle.?
    # use for defining local vs. LS perturbations
    # TODO: easiest way to define would be get mask from pixel level data no?
    #mean_lat = track['meanlat'].values[()]
    #mean_lon = track['meanlon'].values[()]
    #ccs_area = track['ccs_area'].values[()]
    #core_area = track['core_area'].values[()]
    # NOTE: if not all three steps included at start then output is 0
    #total_rain = track['total_rain'].values[()] # aggregated from previous timesteps, no cos(lat) weighting

    return track

# ---------------------------------------------------------------------
def get_months(year):
    if year >= 2018:
        mn1, mn2 = 4, 9
    else:
        mn1, mn2 = 1, 1
    mn1, mn2 = str(mn1).zfill(2), str(mn2).zfill(2)
    return mn1, mn2

# =====================================================================
if __name__ == '__main__':
    main()
