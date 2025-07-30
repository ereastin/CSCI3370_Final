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
from functools import partial

STATS = False
NORM = True
ROI = True
FORECAST = False
AGG = False  # aggregate to 6-hourly.. 3-hourly too hard.?
RM_VARS = [] #['U', 'V', 'T', 'QV']
CLASSIFIER = False

## ================================================================================
def main():
    # TODO: cant get LocalCluster/Client to interact with pytorch DataLoader workers...
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)  # have to specify this think it defaults to 4 or smthn
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        # NOTE: if running monthly.. each month LSF + P ~ 5G
        pd = OTPrecipDataset('train', 'inc3d_reana', season='both', weekly=True, shuffle=False)
        t1 = time.time()
        for i, (s, t, tt) in enumerate(pd):
            print(s.shape, t.shape)
            print(tt, flush=True)
            break

        t2 = time.time()

    if STATS: pd.get_stats()
    print('time:', (t2 - t1))

## ================================================================================
class OTPrecipDataset(Dataset):
    def __init__(self, mode, exp, season='both', weekly=False, shuffle=False, perturb_dict={}):
        super(OTPrecipDataset, self).__init__()
        self.mode = mode
        self.exp = exp
        self.season = season
        self.weekly = weekly
        self.shuffle = shuffle
        self.perturb_dict = perturb_dict

        if self.exp == 'inc3d_reana':
            self.n_months = 6
            mnth_offset = 3
        else:
            self.n_months = 8
            mnth_offset = 4

        yrs = list(range(1980, 2021))
        mnths = list(range(mnth_offset, mnth_offset + self.n_months))
        wks = list(range(4))

        if self.weekly:
            t_strs = [(l[0], l[1], l[2]) for l in list(product(yrs, mnths, wks))]
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
                self.t_strs = t_strs[tr:tr + v]
            case 'test':
                #self.t_strs = t_strs[:1]
                self.t_strs = t_strs[tr + v:]
            case 'prep':
                self.t_strs = t_strs
            case 'check':
                self.t_strs = t_strs
            case _:
                print('Mode {mode} is not correct')
                sys.exit(21)

        self.ctr = 0
        if STATS:
            self.t_strs = np.asarray(self.t_strs)
            smpl = int(0.25 * len(self.t_strs))
            print(f'sampling {smpl} of {len(t_strs)} weeks')
            rnd_idx = rng.integers(len(self.t_strs), size=smpl)
            self.t_strs = self.t_strs[rnd_idx]

        # Necessary for composing filename requests
        self.dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.splits = {  # super ugly..
            31: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 25), np.arange(25, 32)],
            30: [np.arange(1, 9), np.arange(9, 17), np.arange(17, 24), np.arange(24, 31)],
            29: [np.arange(1, 9), np.arange(9, 16), np.arange(16, 23), np.arange(23, 30)],
            28: [np.arange(1, 8), np.arange(8, 15), np.arange(15, 22), np.arange(22, 29)],
        }

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

        self.stats_pth = f'./{self.exp}_agg_norm_vars.json' if AGG else f'./{self.exp}_norm_vars.json'
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
        # TODO: forecasting would be handled here when passing files.. 
        # does open_mfdataset() work with just one item list?
        if self.weekly:
            (year, month, week) = self.t_strs[idx] 
            source_rw_pth, target_rw_pth = self._get_filepaths(year, month, week)
        else:
            (year, month), week = self.t_strs[idx], None
            fname = f'{year}_{str(month).zfill(2)}_*.nc'  # this will not work for writing..

        wk_days, _ = self._get_wk_days(year, month, week)
        time_id = {'yr': year, 'mn': month, 'days': wk_days}

        if self.mode == 'prep':
            curr_files = os.listdir(os.path.join(pth.SCRATCH, self.exp))
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

            if source is None or target is None:
                return (None, None, time_id)

            # Trim if missing timesteps (only good for Dec 31 2020 bc of MSWEP?)
            if source.shape[0] != target.shape[0]:
                source, target = self.prune(source, target, time_id)

            # Shuffle within batch
            if self.shuffle:
                shuff_idx = torch.randperm(source.shape[0])
                source = source[shuff_idx]
                target = target[shuff_idx]

            sample = (source, target, time_id)
            return sample

    # -----------------------------------------------------------------------------
    def _get_filepaths(self, year, month, week):
        # MERRA stuff
        source_rw_pth = os.path.join(pth.SCRATCH, self.exp, f'LSF_{year}_{str(month).zfill(2)}_{week}.nc')

        # MSWEP stuff
        if not FORECAST:
            target_rw_pth = [os.path.join(pth.SCRATCH, self.exp, f'P_{year}_{str(month).zfill(2)}_{week}.nc')]
        else:
            if week == 3:
                paths = [f'P_{year}_{str(month).zfill(2)}_{week}.nc', f'P_{year}_{str(month + 1).zfill(2)}_{0}.nc']
            else:
                paths = [f'P_{year}_{str(month).zfill(2)}_{week}.nc', f'P_{year}_{str(month).zfill(2)}_{week + 1}.nc']
            target_rw_pth = [os.path.join(pth.SCRATCH, self.exp, p) for p in paths]

        return source_rw_pth, target_rw_pth

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
    @staticmethod
    def _select_batch(ds, **kwargs):
        return ds.sel(**kwargs)
    
    # TODO: move to utils
    # -----------------------------------------------------------------------------
    @staticmethod
    def _get_merra_nsel(year, month):
        # N is 1-4 (?) 1 for 1980-1991, 2 for 1992-2000, 3 for 2001-2010, 4 for 2011+ 
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

    # TODO: move to utils ??
    # -----------------------------------------------------------------------------
    def _get_wk_days(self, year, month, week=None):
        leap_year = (year % 4 == 0)
        n_days = self.dpm[month - 1]
        n_days += 1 if leap_year and month == 2 else 0
        wk_days = n_days if week == None else self.splits[n_days][week]
        return wk_days, leap_year

    # TODO: move to utils ??
    # -----------------------------------------------------------------------------
    def _get_files_by_time(self, year, month, week):
        wk_days, leap_year = self._get_wk_days(year, month, week)

        # MERRA fname: <>.YYYYMMDD.nc4
        N = self._get_merra_nsel(year, month)
        merra_base = f'MERRA2_{N}.inst3_3d_asm_Np.' + str(year) + str(month).zfill(2)
        merra_fs = [os.path.join(pth.MERRA,  merra_base + str(day).zfill(2) + '.nc4') for day in wk_days]

        # MSWEP fname: YYYYDDD.HH.nc
        d0 = 1 if leap_year and month > 2 else 0
        day_sum = sum(self.dpm[:month - 1])
        doys = [os.path.join(pth.MSWEP, str(year) + str(day_sum + day).zfill(3)) for day in wk_days + d0]
        hrs = ['.' + str(hr).zfill(2) + '.nc' for hr in range(0, 24, 3)]
        mswep_fs = [f[0] + f[1] for f in list(product(doys, hrs))]
        return merra_fs, mswep_fs

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
    @staticmethod
    def _perturb(ds, instructions):
        perturb_var = instructions['var']
        perturb_type = instructions['type']
        abl = instructions['abl']
        scale = instructions['scale']
        # potentially select certain regions for perturbation.?

        plvls = ds.coords['lev'].values[()]
        if abl: plvls = plvls[:6]  # not necessarily ABL really just low-level phen.
        lats, lons = ds.coords['lat'].values[()], ds.coords['lon'].values[()]
        match perturb_type:
            case 'zero':
                # set values to zero (e.g. winds, q are 'reasonable', removing convection/advection, removing moisture)
                # or name as 'assign' and use for all e.g. subsidence everywhere?
                ds[perturb_var].loc[dict(lev=plvls)] = 0
            case 'scale':  # can also use to invert sign
                ds[perturb_var].loc[dict(lev=plvls)] *= scale
            case 'vshear':  # low-level vertical wind shear thought to be important for MCS (Rotunno '88), also LLJs for moisture?
                # remove per-column vertical shear -> set field at each cell to mean of cell across p-lvls
                ds[perturb_var].loc[dict(lat=lats, lon=lons)] = ds[perturb_var].mean(dim='lev')
            case 'hshear':
                # remove horizontal shear -> set field at each p-lvl to mean of p-lvl
                ds[perturb_var].loc[dict(lev=plvls)] = ds[perturb_var].mean(dim=['lat', 'lon'])
            case 'shuffle':
                k = 8  # adjust this for different scale importance
                for i in range(0, len(lats), k):
                    for j in range(0, len(lons), k):
                        for p in range(0, len(plvls)):
                            slat, slon = slice(i, i + k), slice(j, j + k)
                            v = ds[perturb_var].isel(lat=slat, lon=slon, lev=p).values[()].flatten()
                            np.random.shuffle(v)
                            ds[perturb_var].loc[dict(lat=lats[slat], lon=lons[slon], lev=plvls[p])] = v.reshape((k, k))

        return ds

    # -----------------------------------------------------------------------------
    def perturb(self, ds, perturb_dict):
        if perturb_dict == {}: return ds
        for instr in perturb_dict.items():
            ds = self._perturb(ds, instr)
        return ds

    # -----------------------------------------------------------------------------
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
        # NOTE: not sure this is the best way to handle but other options? interpolate below ground?
        dsBF = ds[['T', 'H']].bfill(dim='lev')
        ds = xr.merge([ds0, dsBF])
        ds = self._regrid(ds, regrid_dict)
        ds = ds.compute()
        ds.to_netcdf(out_file, engine='netcdf4')

    # -----------------------------------------------------------------------------
    def read_source(self, in_file):
        ds = xr.open_dataset(in_file, drop_variables=RM_VARS) if self.weekly else xr.open_mfdataset(in_file)

        if AGG:
            ds = ds.resample(time='6h').mean()

        self.perturb(ds, self.perturb_dict)

        if STATS:
            for v in self.merra_vars:
                dav = ds[v]
                self._stats[v + 'mn'].append(dav.mean().data)
                self._stats[v + 'var'].append(dav.var().data)
            self._stats['Nv'].append(dav.count().data)
            return

        if NORM:
            # select for feqer p-levels
            ds = ds.sel(lev=self.sel_p)
            for v in self.merra_vars:
                if v in RM_VARS:
                    continue
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']
        #if ROI:
        #    ds = ds.sel(lat=slice(24, 50.5), lon=slice(-108.75, -83.125))

        data = ds.to_dataarray().to_numpy()
        source = torch.tensor(data).permute(1, 0, 2, 3, 4) # return as (time, var, (lev), lat, lon) 
        return source

    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    def read_target(self, in_file):
        ds = xr.open_mfdataset(in_file)

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
            #pass
            v = 'precipitation'
            ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']
        
        data = ds.to_dataarray().to_numpy()
        target = torch.tensor(data).permute(1, 0, 2, 3)  # return as (time, 1, lat, lon)

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

        return target
 
    ## ============================================================================
    ## UNUSED
    ## ============================================================================
    def _rm_var(self, ds):
        # what about perturbing? add noise etc?
        lev = ds.coords['lev']
        for p in self.p:
            fill = 0.0 if self.zero else ds[self.rm_var].sel(dict(lev=lev[lev == p])).mean()
            ds[self.rm_var].loc[dict(lev=lev[lev == p])] = fill
        return ds

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
