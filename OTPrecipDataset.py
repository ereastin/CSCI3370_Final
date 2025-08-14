import torch
from torch.utils.data import Dataset
import dask
from dask import array
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import xarray as xr
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
from functools import partial

STATS = False
NORM = True
ROI = False
FORECAST = False
AGG = False  # aggregate to 6-hourly.. 3-hourly too hard.?
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

        yrs = list(range(2004, 2021))  # list(range(1980, 2021))
        mnths = list(range(mnth_offset, mnth_offset + self.n_months))
        wks = list(range(4))

        if self.weekly:
            t_strs = [(l[0], l[1], l[2]) for l in list(product(yrs, mnths, wks))]
            # remove weeks without a single MCS
            rm = [(2014, 3, 2), (2004, 4, 1), (2010, 3, 0), (2019, 3, 2), (2018, 3, 3), (2018, 3, 1), (2020, 3, 2), (2015, 3, 2), (2019, 3, 3), (2015, 3, 0), (2018, 4, 2), (2018, 3, 0), (2020, 3, 3), (2019, 3, 0), (2012, 3, 0), (2014, 3, 3), (2018, 4, 3), (2018, 3, 2), (2006, 3, 0), (2014, 3, 1), (2020, 3, 0), (2013, 3, 0), (2019, 3, 1), (2020, 3, 1), (2004, 3, 1), (2007, 4, 1)]
            for t in rm:
                if t in t_strs:
                    t_strs.pop(t_strs.index(t))
        else:  # monthly
            t_strs = [(l[0], l[1]) for l in list(product(yrs, mnths))]

        # fix the seed
        if self.mode != 'prep':
            rng = np.random.default_rng(seed=4207765)
            rng.shuffle(t_strs)

        l = len(t_strs)
        tr, v, te = int(.80 * l), int(.10 * l), int(.10 * l)
        d = l - (tr + v + te)
        if d > 0:
            v += 1
            te += 1

        match mode:
            case 'train':
                self.t_strs = t_strs[:tr]
            case 'val':
                self.t_strs = t_strs[tr:tr + v]
            case 'test':
                self.t_strs = t_strs[tr + v:]
            case 'check':
                self.t_strs = t_strs
            case _:
                print('Mode {mode} is not correct')
                sys.exit(21)

        #LEV = np.array([
        #    1000, 975, 950, 925, 900, 875, 850,
        #    825, 775, 700, 600, 550, 450, 400, 350, 300,
        #    250, 200, 150, 100, 70, 
        #    50, 40, 30, 20, 10, 7, 3
        #])
        self.p = np.array([
            1000, 975, 950, 925, 900, 875, 850,
            800, 750, 650, 550, 450, 350, 250, 150, 100
        ])


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
        if self.weekly:
            (year, month, week) = self.t_strs[idx] 
            source_rw_pth, target_rw_pth = futils._get_filepaths(year, month, week, self.exp, forecast=FORECAST)
        else:
            (year, month), week = self.t_strs[idx], None
            fname = f'{year}_{str(month).zfill(2)}_*.nc'

        wk_days, _ = futils._get_wk_days(year, month, week)
        time_id = {'yr': year, 'mn': month, 'days': wk_days}

        # NEW filter for MCSs only
        t_slc = mcs.get_times(time_id)
        #t_slc = slice(None, None)

        source = self.read_source(
            source_rw_pth,
            t_slc
            #self.merra_drop_vars,
            #time=t_slc,
            #lev=self.p,
        )
        target = self.read_target(
            target_rw_pth,
            t_slc
            #self.mswep_drop_vars,
            #time=t_slc
        )

        if self.ret_as_tnsr:
            source, target = self._ret_tensor(source, target)
        
        sample = (source, target, time_id)
        return sample

    # -----------------------------------------------------------------------------
    def _ret_tensor(self, source_ds, target_ds):
        # TODO: sure permute is ok? dont need to make sure ordering is correct? what if inverted or something?
        source = torch.tensor(source_ds.to_dataarray().to_numpy())
        source_ds.close()
        if TWOD:
            s = source.shape
            source = source.reshape(s[1], -1, s[3], s[4])
            source = source.permute(1, 0, 2, 3)  # return as (time, var * lev, lat, lon)
        else:
            source = source.permute(1, 0, 2, 3, 4)  # return as (time, var, lev, lat, lon)

        target = torch.tensor(target_ds.to_dataarray().to_numpy())
        target_ds.close()
        target = target.permute(1, 0, 2, 3)  # return as (time, 1[precip], lat, lon)

        # Shuffle within batch
        if self.shuffle:
            shuff_idx = torch.randperm(source.shape[0])
            source = source[shuff_idx]
            target = target[shuff_idx]

        return source, target

    # -----------------------------------------------------------------------------
    @staticmethod
    def prune(source, target, f_id):
        # if batch size doesn't match then trim to smaller
        print(f'pruning {f_id}...')
        trim = min(source.shape[0], target.shape[0])
        return source[:trim], target[:trim]
 
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
    def read_source(self, in_file, t_slc): #drop_vars, **kwargs):
        #preproc_fn = partial(futils._select_batch, **kwargs)
        ds = xr.open_mfdataset(
            in_file
        )
        '''
            ,
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
        '''
        ds = ds.sel(time=t_slc)

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

        return ds

    # -----------------------------------------------------------------------------
    def read_target(self, in_file, t_slc): #drop_vars, **kwargs):
        #preproc_fn = partial(futils._select_batch, **kwargs)
        ds = xr.open_mfdataset(
            in_file
        )
        '''
            ,
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
        '''
        ds = ds.sel(time=t_slc)

        if FORECAST:
            # TODO: fix this to accept MCS time stuff
            # what if want farther out? cant feed LSF autoregressively so would just be a 'jump'
            ds = ds.isel(time=slice(1, 64 + 1))

        if AGG:
            ds = ds.resample(time='6h').sum()
 
        if STATS:
            v = 'precipitation'
            dav = ds[v]
            self._stats[v + 'mn'].append(dav.mean().data)
            self._stats[v + 'var'].append(dav.var().data)
            self._stats['Np'].append(dav.count().data)
            return

        if NORM:
            v = 'precipitation'
            ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']
        
        return ds

## ================================================================================
if __name__ == '__main__':
    main()
