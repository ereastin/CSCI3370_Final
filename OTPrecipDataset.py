import torch
from torch.utils.data import Dataset
import dask
from dask import array
from dask.distributed import Client, LocalCluster
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('/home/eastinev/AI')
import paths as pth
import time
import json
from itertools import product
import mcs_data as mcs
import file_utils as futils

# probably move some of these to class attrs.?
STATS = False
FORECAST = True
if FORECAST:
    LEAD_TIME_IDX = 1
else:
    LEAD_TIME_IDX = 0
AGG = False  # aggregate to 6-hourly.. 3-hourly too hard.?
TWOD = False
DRY = False
MCS = False
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
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)  # have to specify this think it defaults to 4 or smthn
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        pd = OTPrecipDataset('train', 'cus', season='sum', standardize=False, shuffle=False, ret_as_tnsr=True)
        t1 = time.time()
        for i, (s, t, tt) in enumerate(pd):
            if int(tt[0].astype(str).split('-')[0]) == 2012:
                print(1)

    if STATS: pd.get_stats()
    t2 = time.time()
    print('time:', (t2 - t1))

## ================================================================================
class OTPrecipDataset(Dataset):
    def __init__(self, mode, exp, season='both', standardize=True, shuffle=False, ret_as_tnsr=True, cesm=False, cesm_exp=''):
        super(OTPrecipDataset, self).__init__()
        self.mode = mode
        self.exp = exp
        self.season = season
        self._STANDARDIZE = standardize
        self.shuffle = shuffle
        self.ret_as_tnsr = ret_as_tnsr
        self._CESM = cesm
        self._CESM_EXP = cesm_exp

        # TODO: can pull this from file_utils?
        self.n_months = 6 if season == 'both' else 3
        mnth_offset = 6 if season == 'sum' else 3

        yrs = list(range(2004, 2021))  # list(range(1980, 2021))
        mnths = list(range(mnth_offset, mnth_offset + self.n_months))
        wks = list(range(4))

        t_strs = [(l[0], l[1], l[2]) for l in list(product(yrs, mnths, wks))]
        # remove weeks without a single MCS
        #if MCS:
        if not self._CESM:
            for t in RM_WEEKS:
                if t in t_strs:
                    t_strs.pop(t_strs.index(t))
        
        # TODO: fix all this crap
        if self._CESM:
            yrs = list(range(1979, 1984))
            mnths = list(range(mnth_offset, mnth_offset + self.n_months))
            wks = list(range(4))
            cesm_t_strs = [(l[0], l[1], l[2]) for l in list(product(yrs, mnths, wks))]

        # fix the seed
        # TODO: bad way to handle this.. if removing/adding weeks you screw up the shuffle order and can contaminate test data
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
                if MCS and self.season == 'spr' and DRY:
                    self.t_strs += RM_WEEKS  # test these on non-MCS weeks
                if self._CESM:
                    self.t_strs = cesm_t_strs
            case 'check':
                self.t_strs = t_strs
            case _:
                print('Mode {mode} is not correct')
                sys.exit(21)

        ## For variables statistics/data norm
        self.merra_vars = ['U', 'V', 'OMEGA', 'H', 'T', 'QV']  # if interested could use cloud ice and liquid mass mixing ratios?
        self._stats = {k + 'mn': [] for k in self.merra_vars} | {k + 'var': [] for k in self.merra_vars}
        self._stats['Nv'] = []
        self._stats['precipitationmn'] = []
        self._stats['precipitationvar'] = []
        self._stats['Np'] = []

        # TODO: fix this to accept exp name
        # can we add which stats file was used into the hyperparams for loading.?
        # still need to generate by if mode == 'test' then look at hps?
        self.stats_pth = f'./cus_norm_vars_{self.season}'
        if AGG: self.stats_pth += '_agg'
        self.stats_pth += '_mcs'
        self.stats_pth += '.json'
        if os.path.exists(self.stats_pth):
            print(f'Loading training set statistics from {self.stats_pth}')
            with open(self.stats_pth, 'r') as f:
                self.stats = json.load(f)
                # for using CESM data
                self.stats['Z3mn'] = self.stats['Hmn']
                self.stats['Z3std'] = self.stats['Hstd']
                self.stats['Qmn'] = self.stats['QVmn']
                self.stats['Qstd'] = self.stats['QVstd']
                self.stats['PRECTmn'] = self.stats['precipitationmn']
                self.stats['PRECTstd'] = self.stats['precipitationstd']

    # -----------------------------------------------------------------------------
    def __len__(self):
        return len(self.t_strs)

    # -----------------------------------------------------------------------------
    def __getitem__(self, idx):
        (year, month, week) = self.t_strs[idx] 
        source_rw_pth, target_rw_pth = futils._get_filepaths(year, month, week, self.exp, forecast=FORECAST, cesm_exp=self._CESM_EXP)

        # now datetime arange
        if self._CESM:
            time_id = futils._get_wk_days_no_leap(year, month, week)
        else:
            time_id = futils._get_wk_days_leap(year, month, week)

        # Filter for MCSs only
        if MCS:
            t = mcs.get_times(time_id)  # this returns empty list if no MCS present:
            # if .sel(time=[]) this selects NONE, if drop_sel(time=[]) this drops NONE
            if FORECAST:
                t_shift = t + np.timedelta64(3 * LEAD_TIME_IDX, 'h')
            else:
                t_shift = t
        else:
            t, t_shift = slice(None, None), slice(None, None)

        try:
            # read in source and target datasets
            source = self.read_source(
                source_rw_pth,
                sel_time=t
            )
            target = self.read_target(
                target_rw_pth,
                sel_time=t_shift,
                batch_size=len(source.time),  # these used for forecasting
                lead_time=LEAD_TIME_IDX  # INDEX lead time, e.g. 1 == next 3hourly step, 8 == next day
            )
        except FileNotFoundError:
            return None, None, time_id
        except OSError:
            print(source_rw_pth, target_rw_pth)
            exit()

        if STATS:
            return None, None, time_id

        if self.ret_as_tnsr:
            source, target = self._ret_tensor(source, target)

        sample = (source, target, time_id)
        return sample

    # -----------------------------------------------------------------------------
    def _ret_tensor(self, source_ds, target_ds):
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
    def read_source(self, in_file, sel_time=slice(None, None)):
        ds = xr.open_mfdataset(in_file)

        if MCS:
            ds = ds.drop_sel(time=sel_time) if DRY else ds.sel(time=sel_time)

        if AGG:
            ds = ds.resample(time='6h').mean()

        if STATS:
            for v in self.merra_vars:
                dav = ds[v]
                self._stats[v + 'mn'].append(dav.mean().data)
                self._stats[v + 'var'].append(dav.var().data)
            self._stats['Nv'].append(dav.count().data)
            return

        if self._STANDARDIZE:
            for v in ds.variables:
                if v in ['time', 'lat', 'lon', 'plev', 'lev']:
                    continue
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']

        return ds

    # -----------------------------------------------------------------------------
    def read_target(self, in_file, sel_time=slice(None, None), batch_size=None, lead_time=0):
        ds = xr.open_mfdataset(in_file)

        if MCS:
            ds = ds.drop_sel(time=sel_time) if DRY else ds.sel(time=sel_time)

        if FORECAST:  # AFAIK this wont work well for MCS-specific times
            ds = ds.isel(time=slice(lead_time, batch_size + lead_time))

        if AGG:
            ds = ds.resample(time='6h').sum()
 
        if STATS:
            v = 'precipitation'
            dav = ds[v]
            self._stats[v + 'mn'].append(dav.mean().data)
            self._stats[v + 'var'].append(dav.var().data)
            self._stats['Np'].append(dav.count().data)
            return

        if self._STANDARDIZE:
            for v in ds.variables:
                if v in ['time', 'lat', 'lon', 'plev', 'lev']:
                    continue
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']
        
        return ds

## ================================================================================
if __name__ == '__main__':
    main()
