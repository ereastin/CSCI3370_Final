import torch
from torch.utils.data import Dataset
import dask
from dask import array
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import xarray as xr  # faster resampling with 'flox' installed?
from virtualizarr import open_virtual_dataset
import numpy as np
import os
import glob
import sys
sys.path.append('/home/eastinev/AI')
import paths as pth
import time
from itertools import product
from functools import partial

STATS = False
NORM = True
ROI = False
FORECAST = False

'''
# NOTE: this is actually way slower, takes ~1 min w/32 cores to load a given sample using time slicing from mfdataset
tt = time.time()
self.source = self.new_get_source(
    self.merra_drop_vars,
    self.merra_regrid,
    lat=self.merra_lats,
    lon=self.merra_lons,
    lev=self.p,
)
tz = time.time()
print(f'? {tz - tt}')
#self.source = self.source_ds.copy()
'''
## ================================================================================
def main():
    # think then just need to init this in train_model.?
    # or where does this go? can this house the Client/LocalCluster?
    # how does that interact with the pytorch num_workers and prefetch stuff?
    # sitting at ~3-4 seconds per iteration with 32 cores at only 500MB each
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)  # have to specify this think it defaults to 4 or smthn
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        pd = OTPrecipDataset('check', 'inc3d_ss', season='both', weekly=True)
        t1 = time.time()
        for i, (s, t, tt) in enumerate(pd):
            #print(tt)
            pass
        # TODO: so this works BUT lon/lat def -180:180/90:-90 so either shift or pay attention to slice order
        # also mswep stuff is in 'center' of 0.1x0.1 square.? so shift by 0.05 in interp?
        t2 = time.time()

    if STATS: pd.get_stats()  # TODO: was this ever run for random weekly stuff
    print('average time per batch:', (t2 - t1) / (i + 1))

## ================================================================================
class OTPrecipDataset(Dataset):
    def __init__(self, mode, exp, season='both', weekly=False, rm_var=None, zero=False):
        super(OTPrecipDataset, self).__init__()
        self.season = season
        self.n_months = 8 # gonna do apr-nov, most common sumatra squall months # 6 if season == 'both' else 3
        yrs = list(range(1980, 2021))
        mnth_offset = 4 # 6 if season == 'sum' else 3
        mnths = list(range(mnth_offset, mnth_offset + self.n_months))
        wks = list(range(4))
        t_strs = np.asarray([(l[0], l[1], l[2]) for l in list(product(yrs, mnths, wks))])
        # fix the seed, at least one shuffle is the same everytime AFAIK..
        rng = np.random.default_rng(seed=4207765)
        for _ in range(3):
            rng.shuffle(t_strs)

        l = len(t_strs)
        tr, v, te = int(.80 * l), int(.10 * l), int(.10 * l)
        d = l - (tr + v + te)
        if d > 0:
            v += 1
            te += 1
        # NOTE: these may not add to len(t_strs) but should cover dataset bc te not used

        match mode:
            case 'train':
                self.t_strs = t_strs[:tr]
            case 'val':
                self.t_strs = t_strs[tr:tr + v]
            case 'test':
                self.t_strs = t_strs[tr + v:]
            case 'prep':
                self.t_strs = t_strs
            case 'check':
                self.t_strs = t_strs
            case _:
                print('Mode {mode} is not correct')
                sys.exit(21)

        if STATS:
            rnd_idx = rng.integers(len(self.t_strs), size=100)
            self.t_strs = self.t_strs[rnd_idx]
            print(self.t_strs, len(self.t_strs))

        self.mode = mode
        self.exp = exp
        self.weekly = weekly
        self.rm_var = rm_var
        self.zero = zero
        self._stats = {
            'QVmin': [], 'QVmax': [], 'QVmn': [], 'QVstd': [],
            'Tmin': [], 'Tmax': [], 'Tmn': [], 'Tstd': [],
            'Umin': [], 'Umax': [], 'Umn': [], 'Ustd': [],
            'Vmin': [], 'Vmax': [], 'Vmn': [], 'Vstd': [],
            'OMEGAmin': [], 'OMEGAmax': [], 'OMEGAmn': [], 'OMEGAstd': [],
            'Hmin': [], 'Hmax': [], 'Hmn': [], 'Hstd': [],
        }
        # NOTE: these stats for sumatra squall stuff
        self.stats = {
            'QVmn': 4.53e-3, 'QVstd': 5.35e-3,
            'Tmn': 261.68, 'Tstd': 32.009,
            'Umn': -3.58, 'Ustd': 11.304,
            'Vmn': 0.1221, 'Vstd': 3.989,
            'OMEGAmn': -0.01125, 'OMEGAstd': 0.1245,
            'Hmn': 10997.52, 'Hstd': 11419.04,
        }

        # Necessary for composing filename requests
        self.dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        # TODO: my god this is ugly... better way?
        self.splits = {
            31: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 25), np.arange(25, 32)],
            30: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 24), np.arange(24, 31)],
            29: [np.arange(1, 9), np.arange(9, 16), np.arange(16, 23), np.arange(23, 30)],
            28: [np.arange(1, 8), np.arange(8, 15), np.arange(15, 22), np.arange(22, 29)],
        }

        # MERRA selection criteria
        self.merra_regrid = {'do': False, 'extent': None, 'steps': None}
        # self.merra_lats = slice(14, 53.5)
        # self.merra_lons = slice(-140, -50.625)
        self.merra_lats = slice(-15.5, 24)
        self.merra_lons = slice(55, 144.625)
        self.merra_drop_vars = ['EPV', 'SLP', 'PS', 'RH', 'PHIS', 'O3', 'QI', 'QL']
        self.merra_vars = ['U', 'V', 'OMEGA', 'H', 'T', 'QV']  # if interested could use cloud ice and liquid mass mixing ratios?
        self.p = np.array([
            1000, 975, 950, 925, 900, 875, 850,
            825, 800, 775, 750, 725, 700, 650,
            600, 550, 500, 450, 400, 350, 300,
            250, 200, 150, 100, 70, 50, 40,
            30, 20, 10, 7, 5, 4, 3
        ])

        # MSWEP selection criteria
        self.mswep_regrid = {'do': False, 'extent': None, 'steps': None}
        # self.mswep_regrid = {'do': True, 'extent': (24, 50.5, -108.75, -83.125), 'steps': (0.5, 0.625)}
        # self.mswep_lats = slice(50.5, 24)
        # self.mswep_lons = slice(-108.75, -83.125)
        self.mswep_lats = slice(8.0, 0.0)
        self.mswep_lons = slice(92.8, 107.2)
        self.mswep_drop_vars = []

    def __len__(self):
        if self.weekly:
            return len(self.t_strs)
        else:  # assume monthly
            return self.n_months * len(self.years) # 12 * len(self.years)

    def __getitem__(self, idx):
        """
        select batch by idx
        """
        (year, month, week) = self.t_strs[idx]
        fname = f'{year}_{str(month).zfill(2)}_{week}.nc'
        wk_days, _ = self._get_wk_days(year, month, week)
        time_id = {'yr': year, 'mn': month, 'days': wk_days}
        source_rw_pth = os.path.join(pth.SCRATCH, self.exp, 'LSF_' + fname)
        target_rw_pth = os.path.join(pth.SCRATCH, self.exp, 'P_' + fname)

        if self.mode == 'prep':
            curr_files = os.listdir(os.path.join(pth.SCRATCH, self.exp))
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
            merra_files, mswep_files = self._get_files_by_time(year, month, week)
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
        else:
            try:
                source = self.read_source(
                    source_rw_pth
                )
            except FileNotFoundError as e:
                print(f'MERRA {source_rw_pth} failed, DNE', e)
                source = None
            try:
                target = self.read_target(
                    target_rw_pth
                )
            except FileNotFoundError as e:
                print(f'MSWEP {target_rw_pth} failed, DNE', e)
                target = None
            #if source.shape[0] != target.shape[0]:
            #    source, target = self.prune(source, target, fname)
            sample = (source, target, time_id)
            return sample

        '''
        #see note above re: opening entire dataset
        tslc = slice(f'{year}-{month}-{week * 7 + 1}', f'{year}-{month}-{week * 7 + 7 + 1}')
        source = self.source_gettr(
            time_slc=tslc,  # this comes from the index? easier to use a map instead of iter.?
        )
        #this is just as slow in serial anyway
        self.vzarr_get_source(
            fname,
            merra_files,
            self.merra_drop_vars,
        )
        source = self.vzarr_load_source(
            fname,
            lat=self.merra_lats,
            lon=self.merra_lons,
            lev=self.p
        )
        '''

    @staticmethod
    def prune(source, target, f_id):
        # if batch size doesn't match then trim to smaller
        print(f'pruning {f_id}...')
        trim = min(source.shape[0], target.shape[0])
        return source[:trim], target[:trim]

    @staticmethod
    def _regrid(ds, regrid_dict):
        if not regrid_dict['do']: return ds
        extent = regrid_dict['extent']
        steps = regrid_dict['steps']
        # TODO: if extent is None use current bounds
        lat1, lat2 = min(extent[0], extent[1]), max(extent[0], extent[1]) #round(ds.lat[0].item(), 2), round(ds.lat[-1].item(), 2)
        lon1, lon2 = min(extent[2], extent[3]), max(extent[2], extent[3]) #round(ds.lon[0].item(), 2), round(ds.lon[-1].item(), 2)
        # again pay attention to order of values
        new_lat = np.arange(lat1, lat2 + steps[0], steps[0])
        new_lon = np.arange(lon1, lon2 + steps[1], steps[1])
        # TODO: check other interp methods
        return ds.interp(lat=new_lat, lon=new_lon)

    @staticmethod
    def _select_batch(ds, **kwargs):
        return ds.sel(**kwargs)

    @staticmethod
    def _get_merra_nsel(year):
        # TODO: apparently 1980-1984 and 2009+ removed..?
        # N is 1-4 (?) 1 for 1980-1991, 2 for 1992-2000, 3 for 2001-2010, 4 for 2011+ 
        # gotta be faster way
        if year in list(range(1980, 1992)):
            N = 1
        elif year in list(range(1992, 2001)):
            N = 2
        elif year in list(range(2001, 2011)):
            N = 3
        else:
            N = 4
        return N

    def _get_wk_days(self, year, month, week):
        leap_year = (year % 4 == 0)
        n_days = self.dpm[month - 1]
        n_days += 1 if leap_year and month == 2 else 0
        wk_days = self.splits[n_days][week]
        return wk_days, leap_year

    def _get_files_by_time(self, year, month, week):
        # TODO: forecasting would need opening the 'week' +- a couple days dep on how far out
        # -> create xr Dataset then .sel() specific time slice
        # opening entire dataset and reusing ds ref is way too slow
        wk_days, leap_year = self._get_wk_days(year, month, week)

        # NOTE: glob is crazy slow in these dirs bc theyre huge so just build fnames directly
        N = self._get_merra_nsel(year)
        merra_base = f'MERRA2_{N}00.inst3_3d_asm_Np.' + str(year) + str(month).zfill(2)
        merra_fs = [os.path.join(pth.MERRA,  merra_base + str(day).zfill(2) + '.nc4') for day in wk_days]

        # TODO: forecasting addition would go here
        # really gonna a need a 'streaming' version of this otherwise need to rewrite the local files every fckn time...
        # depends how far forward we want to forecast no? next 3 hours? 
        # if forecasting more than one timestep how does that work?
        # want it so that model predicts each step or just outright predicts N hours/days in the future?
        # this model type isnt predicting the large scale fields, so probs not predict each step and feed back in
        # NOTE: for this just do open_mfdataset that encompasses and then slice to appropriate times
        if FORECAST:  # easiest here is 1 day
            # if just one day still need the extra day.. easiest to break by hours then
            forecast_reach = 3  # hours
            forecast_day = forecast_reach // 24
            forecast_hr = forecast_reach % 24
        else:
            forecast_day = 0
            forecast_hour = 0

        # MSWEP fname: YYYYDDD.HH.nc
        d0 = 1 if leap_year and month > 2 else 0
        day_sum = sum(self.dpm[:month - 1])
        doys = [os.path.join(pth.MSWEP, str(year) + str(day_sum + day).zfill(3)) for day in wk_days + d0]
        hrs = ['.' + str(hr).zfill(2) + '.nc' for hr in range(0, 24, 3)]
        mswep_fs = [f[0] + f[1] for f in list(product(doys, hrs))]
        return merra_fs, mswep_fs

    def get_stats(self):
        for v in self.merra_vars:
            vm, vs = np.mean(self._stats[v + 'mn']), np.mean(self._stats[v + 'std'])
            mx, mn = np.amax(self._stats[v + 'max']), np.amin(self._stats[v + 'min'])
            print(f'Variable {v}: mean {vm} std {vs} max {mx} min {mn}')

    def _rm_var(self, ds):
        # what about perturbing? add noise etc?
        lev = ds.coords['lev']
        for p in self.p:
            fill = 0.0 if self.zero else ds[self.rm_var].sel(dict(lev=lev[lev == p])).mean()
            ds[self.rm_var].loc[dict(lev=lev[lev == p])] = fill
        return ds

    def write_source(self, out_file, source_files, drop_vars, regrid_dict, **kwargs):
        preproc_fn = partial(self._select_batch, **kwargs)
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
        dsBF = ds[['T', 'H']].bfill(dim='lev')
        ds = xr.merge([ds0, dsBF])
        ds = self._regrid(ds, regrid_dict)
        ds = ds.compute()
        ds.to_netcdf(out_file, engine='netcdf4')

    def read_source(self, in_file):
        ds = xr.open_dataset(in_file)

        if self.rm_var is not None:
            ds = self._rm_var(ds)

        if STATS:
            for v in self.merra_vars:
                dav = ds[v]
                self._stats[v + 'max'].append(dav.max().data)
                self._stats[v + 'min'].append(dav.min().data)
                self._stats[v + 'mn'].append(dav.mean().data)
                self._stats[v + 'std'].append(dav.std().data)
            return

        if NORM:
            for v in self.merra_vars:
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']

        #if sum(ds.isnull().sum().to_dataarray().values[()]) > 0:
        #    print('merra', in_file)
        #return
        data = ds.to_dataarray().to_numpy()
        source = torch.tensor(data).permute(1, 0, 2, 3, 4) # return as (time, var, (lev), lat, lon) 
        return source

    def write_target(self, out_file, target_files, drop_vars, regrid_dict, **kwargs):
        preproc_fn = partial(self._select_batch, **kwargs)
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

    def read_target(self, in_file):
        ds = xr.open_dataset(in_file)
        if ROI:
            pass # add this back in
        if STATS:
            # TODO: impl something here to identify 'weeks' with high amount of precip (by index) and reuse them.. somehow
            pass

        #if sum(ds.isnull().sum().to_dataarray().values[()]) > 0:
        #    print('mswep', in_file)
        #return
        data = ds.to_dataarray().to_numpy()
        target = torch.tensor(data).permute(1, 0, 2, 3)  # return as (time, 1, lat, lon)
        return target
    
    ## ============================================================================
    ## UNUSED
    ## ============================================================================
    def close_distr(self):
        self.client.close()
        self.cluster.close()

    def source_gettr(self, time_slc=None):
        new_ds = self.source.sel(time=time_slc)
        data = new_ds.to_dataarray().to_numpy()
        return torch.tensor(data).permute(1, 0, 2, 3, 4) # return as (time, var, (lev), lat, lon) 

    def new_get_source(self, drop_vars, regrid_dict, **kwargs):
        preproc_fn = partial(self._select_batch, **kwargs)
        ds = xr.open_mfdataset(
            os.path.join(pth.MERRA, '*.nc4'),
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
        return ds

    # NOTE: vzarr garbage is literally just as slow and can't grab region before using xr.open_dataset()..
    def vzarr_get_source(self, fname, source_files, drop_vars):
        virtual_datasets = [
            open_virtual_dataset(filepath, drop_variables=drop_vars)
            for filepath in source_files
        ]

        # this Dataset wraps a bunch of virtual ManifestArray objects directly
        virtual_ds = xr.combine_nested(virtual_datasets, concat_dim='time', data_vars='minimal', coords='minimal', compat='override', join='override')

        # cache the combined dataset pattern to disk, in this case using the existing kerchunk specification for reference files
        virtual_ds.virtualize.to_kerchunk(os.path.join(pth.SCRATCH, f'{fname}.json'), format='json')

    def vzarr_load_source(self, fname, **kwargs):
        ds = xr.open_dataset(os.path.join(pth.SCRATCH, f'{fname}.json'), engine='kerchunk')
        ds = ds.sel(**kwargs)
        data = ds.to_dataarray().to_numpy()
        return torch.tensor(data).permute(1, 0, 2, 3, 4) # return as (time, var, (lev), lat, lon) 
    ## ============================================================================
    ## ============================================================================

## ================================================================================
if __name__ == '__main__':
    main()
