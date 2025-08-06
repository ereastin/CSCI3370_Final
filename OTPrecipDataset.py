import torch
from torch.utils.data import Dataset
import dask
from dask import array
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import xarray as xr
import xarray_regrid
#from virtualizarr import open_virtual_dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys
sys.path.append('/home/eastinev/AI')
import paths as pth
import time
from scipy.stats import boxcox
import json
from itertools import product
import mcs_data as mcs
import file_utils as futils

STATS = False
NORM = True
ROI = False
FORECAST = False
AGG = False  # aggregate to 6-hourly.. 3-hourly too hard.?
CLASSIFIER = False
RET_AS_TNSR = True
TWOD = False

## ================================================================================
def main():
    # TODO: cant get LocalCluster/Client to interact with pytorch DataLoader workers...
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)  # have to specify this think it defaults to 4 or smthn
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        # NOTE: if running monthly.. each month LSF + P ~ 5G
        pd = OTPrecipDataset('train', 'inc3d_reana', season='sum', weekly=True, shuffle=False, ret_as_tnsr=RET_AS_TNSR)
        t1 = time.time()
        for i, (s, t, tt) in enumerate(pd):
            print(tt)

        t2 = time.time()

    if STATS: pd.get_stats()
    print('time:', (t2 - t1))

## ================================================================================
class OTPrecipDataset(Dataset):
    def __init__(self, mode, exp, season='both', weekly=False, shuffle=False, ret_as_tnsr=True):
        super(OTPrecipDataset, self).__init__()
        self.mode = mode
        self.exp = exp
        self.season = season
        self.weekly = weekly
        self.shuffle = shuffle
        self.ret_as_tnsr = ret_as_tnsr

        if self.exp == 'inc3d_reana':
            self.n_months = 6 if season == 'both' else 3
            mnth_offset = 6 if season == 'sum' else 3
        else:
            self.n_months = 8
            mnth_offset = 4

        yrs = list(range(2004, 2021)) #list(range(1980, 2021))
        mnths = list(range(mnth_offset, mnth_offset + self.n_months))
        wks = list(range(4))

        if self.weekly:
            t_strs = [(l[0], l[1], l[2]) for l in list(product(yrs, mnths, wks))]
            # remove weeks without a single MCS
            rm = [(2010, 3, 0), (2019, 3, 2), (2018, 3, 3), (2018, 3, 1), (2020, 3, 2), (2015, 3, 2), (2019, 3, 3), (2015, 3, 0), (2018, 4, 2), (2018, 3, 0), (2020, 3, 3), (2019, 3, 0), (2012, 3, 0), (2014, 3, 3), (2018, 4, 3), (2018, 3, 2), (2006, 3, 0), (2014, 3, 1), (2020, 3, 0), (2013, 3, 0), (2019, 3, 1), (2020, 3, 1), (2004, 3, 1), (2007, 4, 1)]
            for t in rm:
                if t in t_strs:
                    t_strs.pop(t_strs.index(t))
        else:  # monthly
            t_strs = [(l[0], l[1]) for l in list(product(yrs, mnths))]

        # fix the seed, at least one shuffle is the same everytime AFAIK..
        if self.mode != 'prep':
            rng = np.random.default_rng(seed=4207765)
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
                #self.t_strs = t_strs[:1]
                self.t_strs = t_strs[:tr]
            case 'val':
                #self.t_strs = t_strs[1:2]
                self.t_strs = t_strs[tr:tr + v]
            case 'test':
                #self.t_strs = t_strs[:1]
                self.t_strs = t_strs[tr + v:]
            case 'check':
                self.t_strs = t_strs
            case _:
                print('Mode {mode} is not correct')
                sys.exit(21)

        self.ctr = 0  # ??

        # MERRA selection criteria
        self.merra_regrid = {'do': False, 'extent': None, 'steps': None}
        # CUS
        self.merra_lats = slice(14, 53.5)
        self.merra_lons = slice(-140, -50.625)
        # sumatra squall
        #self.merra_lats = slice(-15.5, 24)
        #self.merra_lons = slice(55, 144.625)
        self.merra_drop_vars = ['EPV', 'SLP', 'PS', 'RH', 'PHIS', 'O3', 'QI', 'QL'] # NOTE: can use RH instead of Q..
        self.merra_vars = ['U', 'V', 'OMEGA', 'H', 'T', 'QV']  # if interested could use cloud ice and liquid mass mixing ratios?
        #self.p = np.array([
        #    1000, 975, 950, 925, 900, 875, 850,
        #    825, 800, 775, 750, 725, 700, 650,
        #    600, 550, 500, 450, 400, 350, 300,
        #    250, 200, 150, 100, 70, 50, 40,
        #    30, 20, 10, 7, 5, 4, 3
        #])
        # from zhang repr... try just using these 8/similar.? maybe just too many features
        self.sel_p = np.array([1000, 975, 925, 850, 750, 550, 450, 250])
        self.p = np.array([
            1000, 975, 950, 925, 900, 875, 850,
            800, 750, 650, 550, 450, 350, 250, 150, 100
        ])

        # MSWEP selection criteria
        self.mswep_regrid = {'do': True, 'extent': (24, 50.5, -108.75, -83.125), 'steps': (0.5, 0.625)}
        buff = 1
        self.mswep_lats = slice(50.5 + buff, 24 - buff)
        self.mswep_lons = slice(-108.75 - buff, -83.125 + buff)
        # sumatra squall
        #self.mswep_regrid = {'do': False, 'extent': None, 'steps': None}
        #self.mswep_lats = slice(8.0, 0.0)
        #self.mswep_lons = slice(92.8, 107.2)
        self.mswep_drop_vars = []

        ## For variables statistics/data norm
        self._stats = {k + 'mn': [] for k in self.merra_vars} | {k + 'var': [] for k in self.merra_vars}
        self._stats['Nv'] = []
        self._stats['precipitationmn'] = []
        self._stats['precipitationvar'] = []
        self._stats['Np'] = []

        self.stats_pth = f'./{self.exp}_norm_vars_{self.season}'
        if AGG: self.stats_pth += '_agg'
        if ROI: self.stats_pth += '_roi'
        self.stats_pth += '_mcs'
        self.stats_pth += '.json'
        if os.path.exists(self.stats_pth):
            with open(self.stats_pth, 'r') as f:
                self.stats = json.load(f)

    # -----------------------------------------------------------------------------
    def __len__(self):
        return len(self.t_strs)

    # -----------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        select batch by idx
        """
        if self.weekly:
            (year, month, week) = self.t_strs[idx] 
            source_rw_pth, target_rw_pth = futils._get_filepaths(year, month, week, self.exp, forecast=FORECAST)
        else:
            (year, month), week = self.t_strs[idx], None
            fname = f'{year}_{str(month).zfill(2)}_*.nc'  # this will not work for writing..

        wk_days, _ = futils._get_wk_days(year, month, week)
        time_id = {'yr': year, 'mn': month, 'days': wk_days}
        # NEW filter for MCSs only
        t_slc = mcs.get_times(time_id) # should never be empty list

        # NOTE: shoudn't need try-catch here..
        try:
            source = self.read_source(
                source_rw_pth,
                t_slc
            )
        except FileNotFoundError as e:
            print(f'MERRA {source_rw_pth} failed, DNE', e)
            source = None
        try:
            target = self.read_target(
                target_rw_pth,
                t_slc
            )
        except FileNotFoundError as e:
            print(f'MSWEP {target_rw_pth} failed, DNE', e)
            target = None

        if source is None or target is None:
            return (None, None, time_id)

        # Trim if missing timesteps (only good for Dec 31 2020 bc of MSWEP?)
        #if source.shape[0] != target.shape[0]:
        #    source, target = self.prune(source, target, time_id)

        # Shuffle within batch
        if self.shuffle:
            shuff_idx = torch.randperm(source.shape[0])
            source = source[shuff_idx]
            target = target[shuff_idx]

        sample = (source, target, time_id)
        return sample

    # -----------------------------------------------------------------------------
    @staticmethod
    def prune(source, target, f_id):
        # if batch size doesn't match then trim to smaller
        print(f'pruning {f_id}...')
        trim = min(source.shape[0], target.shape[0])
        return source[:trim], target[:trim]

    # -----------------------------------------------------------------------------
    @staticmethod
    def _regrid(ds, regrid_dict):
        if not regrid_dict['do']: return ds
        ds_grid = xr.open_dataset('./pgrid.nc')
        ds = ds.regrid.conservative(ds_grid)
        return ds
 
    # -----------------------------------------------------------------------------
    def get_stats(self):
        print('writing var stats...')
        with open(self.stats_pth, 'w') as f:
            Nv = np.asarray(self._stats.pop('Nv'))
            Np = np.asarray(self._stats.pop('Np'))
            stats_dump = {}
            for v in self.merra_vars:
                mn, vr = np.asarray(self._stats[v + 'mn']), np.asarray(self._stats[v + 'var'])
                comb_mn = np.sum(Nv * mn) / np.sum(Nv)
                stats_dump[v + 'mn'] = comb_mn
                qc = np.sum((Nv - 1) * vr + Nv * mn ** 2)
                stats_dump[v + 'std'] = np.sqrt((qc - np.sum(Nv) * comb_mn ** 2) / (np.sum(Nv) - 1))

            mn, vr = np.asarray(self._stats['precipitationmn']), np.asarray(self._stats['precipitationvar'])
            comb_mn = np.sum(Np * mn) / np.sum(Np)
            stats_dump['precipitationmn'] = comb_mn
            qc = np.sum((Np - 1) * vr + Np * mn ** 2)
            stats_dump['precipitationstd'] = np.sqrt((qc - np.sum(Np) * comb_mn ** 2) / (np.sum(Np) - 1))

            json.dump(stats_dump, f)

    # -----------------------------------------------------------------------------
    def read_source(self, in_file, t_slc):
        ds = xr.open_dataset(in_file) if self.weekly else xr.open_mfdataset(in_file)

        # select for fewer p-levels and MCS times
        ds = ds.sel(time=t_slc)
        if ROI:
            ds = ds.sel(lat=slice(24, 50.5), lon=slice(-108.75, -83.125))

        if AGG:
            ds = ds.resample(time='6h').mean()

        if STATS:
            for v in self.merra_vars:
                dav = ds[v]
                self._stats[v + 'mn'].append(dav.mean().data)
                self._stats[v + 'var'].append(dav.var().data)
            self._stats['Nv'].append(dav.count().data)
            return

        if NORM:
            for v in self.merra_vars:
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']

        if self.ret_as_tnsr:
            data = ds.to_dataarray().to_numpy()
            source = torch.tensor(data)
            if TWOD:
                s = source.shape
                source = source.reshape(s[1], -1, s[3], s[4]) # return as (time, var, (lev), lat, lon) 
            else:
                source = source.permute(1, 0, 2, 3, 4) # return as (time, var, (lev), lat, lon) 
            return source
        else:
            return ds

    # -----------------------------------------------------------------------------
    def read_target(self, in_file, t_slc):
        ds = xr.open_mfdataset(in_file)
        ds = ds.sel(time=t_slc)

        if FORECAST:
            # this should work right -> just forecasting next timestep..
            # what if want farther out? cant feed LSF autoregressively so would just be a 'jump'
            ds = ds.isel(time=slice(1, 64 + 1))

        if AGG:
            ds = ds.resample(time='6h').sum()

        if ROI:
            pass # add this back in
 
        if STATS:
            v = 'precipitation'
            dav = ds[v]
            self._stats[v + 'mn'].append(dav.mean().data)
            self._stats[v + 'var'].append(dav.var().data)
            self._stats['Np'].append(dav.count().data)
            return

        if NORM and not CLASSIFIER:
            v = 'precipitation'
            ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']
        
        if self.ret_as_tnsr:
            data = ds.to_dataarray().to_numpy()
            target = torch.tensor(data).permute(1, 0, 2, 3)  # return as (time, 1, lat, lon)
            return target
        else:
            return ds

        '''
        if CLASSIFIER:
            # NOTE: cleaner way to do this..?
            s = 6 if AGG else 3
            target = torch.where(target > 3, 1, 0)
            #target = torch.where(target == 0, 0, target)
            #target = torch.where((target > 0) & (target < 2.5 * s), 1, target)  # light rain
            #target = torch.where((target >= 2.5 * s) & (target < 7.5 * s), 2, target)  # moderate rain
            #target = torch.where((target >= 7.5 * s) & (target < 50 * s), 3, target)  # heavy rain
            #target = torch.where(target >= 50 * s, 4, target)  # violent rain
            target = target.squeeze(1)
        '''

## ================================================================================
if __name__ == '__main__':
    main()
