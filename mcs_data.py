import xarray as xr
import numpy as np
import os
import sys
sys.path.append('/home/eastinev/AI')

import paths as pth
# =====================================================================
def main():
    read_stats(2004)

# ---------------------------------------------------------------------
def read_data(time_id):
    """
    Filenames:
        ???

    Useful variables:
        [???]
    """
    path = os.path.join(pth.MCS_DATA, '???.nc')
    ds.open_dataset(path)

    # do data read
    return

# ---------------------------------------------------------------------
def read_stats(year):
    """
    Filenames:
        mcs_tracks_final_extc_YYYYMMDD.0000-YYYYMMDD.0000.nc
        -> 2004-2017 all seasons (eg 20170101.0000-20180101.0000)
        -> 2018-2021 warm season (Apr-Sep, eg 20180401.0000-20190901.0000)
    **For full suite of available MCS statistics see README
    Useful stats:
        [track_duration, mcs_duration, start_basetime, end_basetime, meanlat, meanlon, total_rain, conv_rain, strat_rain, lifecycle_stage, lifecycle_index, lifecycle_complete_flag]
    """
    if year >= 2018:
        mn1, mn2 = 4, 9
    else:
        mn1, mn2 = 1, 1
    mn1, mn2 = str(mn1).zfill(2), str(mn2).zfill(2)
    path = os.path.join(pth.MCS_STATS, f'mcs_tracks_final_extc_{year}{mn1}01.0000_{year+1}{mn2}01.0000.nc')
    ds = xr.open_dataset(path)

    #path = os.path.join(pth.MCS_STATS, f'mcs_tracks_final_extc_*.nc')
    #ds = xr.open_mfdataset(path)

    # do stats read
    idx = (ds['lifecycle_complete_flag'] == 1).compute()
    tr1 = ds.isel(tracks=idx)
    print(tr1)
    return

# =====================================================================
if __name__ == '__main__':
    main()
