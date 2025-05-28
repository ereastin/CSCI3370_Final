import torch
from torch.utils.data import Dataset
from dask import array
import xarray as xr
import numpy as np
import os
import sys
sys.path.append('/home/eastinev/AI')
import paths as pth
import time
import matplotlib.pyplot as plt
from itertools import product

DEEP = True
STATS = False
NORM = True
ROI = False

# TODO: worth trying a version of this with open_mfdataset()?
# then could just grab it all at once and select what is needed at runtime
# how is mem handled then tho? is file kept open the whole time? probably.?
# if one week is selected is that array put in mem? how is it reused?
# if literally opening the entire thing how are multiple workers handled? DDP?
# AND what about remapping... if this was just selecting a lat/lon box and
# maybe applying some transforms then sure, but with remap probably too slow?
# how does xarray even handle these? just coarsening MSWEP probably OK?
## ================================================================================
def main():
    t1 = time.time()
    ds = xr.open_mfdataset(os.path.join(pth.MSWEP, '200136*.nc'), chunks={})
    print(ds)
    pd = PrecipDataset('train', 'inc3d_reana', season='both', weekly=True)
    # TODO: so this works BUT lon/lat def -180:180/90:-90 so either shift or pay attention to slice order
    # also mswep stuff is in 'center' of 0.1x0.1 square.? so shift by 0.05 in interp?
    # self.bbox = '-140,-50.625,14,53.5'
    ds1 = pd.select_batch(ds, time=slice('2001-12-27', '2001-12-28'), lat=slice(53.5, 14), lon=slice(-140, -50.625))
    print(ds1)
    ds1 = pd.regrid(ds1, new_lat_step=0.5, new_lon_step=0.625)
    print(ds1)
    #s, t, tt = pd.__getitem__(20)
    #print(s.shape, t.shape)
    #return
    t2 = time.time()
    #pd.get_stats()  # TODO: was this ever run for random weekly stuff
    print(t2 - t1)

## ================================================================================
class PrecipDataset(Dataset):
    def __init__(self, mode, exp, season='both', weekly=False, rm_var=None, zero=False):
        super(PrecipDataset, self).__init__()

        self.season = season
        self.n_months = 6 if self.season == 'both' else 3
        yrs = list(map(str, range(1980, 2021)))
        mnth_offset = 6 if season == 'sum' else 3
        mnths = list(map(lambda x: str(x).zfill(2), range(mnth_offset, mnth_offset + self.n_months)))
        wks = list(map(str, range(4)))
        t_strs = np.asarray([f'{l[0]}{l[1]}_{l[2]}' for l in list(product(yrs, mnths, wks))])
        # fix the seed, at least one shuffle is the same everytime AFAIK..
        rng = np.random.default_rng(seed=4207765)
        for _ in range(3):
            rng.shuffle(t_strs)

        match mode:
            case 'train':
                self.years = [1980, 1983, 1984, 1985, 1986, 1987, 1989, 1990, 1992, 1993, 1995, 1996, 1997, 1998, 1999, 2001, 2003, 2004, 2006, 2007, 2008, 2009, 2010, 2012, 2014, 2015, 2017, 2018, 2020]
                self.t_strs = t_strs[:29 * self.n_months * 4]
            case 'val':
                self.years = [1988, 2019, 2011, 1981, 1994, 2002]
                self.t_strs = t_strs[29 * self.n_months * 4:35 * self.n_months * 4]
            case 'test':
                self.years = [2000, 1982, 2013, 1991, 2005, 2016]
                self.t_strs = t_strs[35 * self.n_months * 4:]
            case _:
                print('Mode {mode} is not correct')
                sys.exit(21)

        self.exp = exp
        self.weekly = weekly
        self.rm_var = rm_var
        self.zero = zero
        self.vars = ['QV', 'T', 'U', 'V', 'OMEGA', 'H']
        self._stats = {
            'QVmin': [], 'QVmax': [], 'QVmn': [], 'QVstd': [],
            'Tmin': [], 'Tmax': [], 'Tmn': [], 'Tstd': [],
            'Umin': [], 'Umax': [], 'Umn': [], 'Ustd': [],
            'Vmin': [], 'Vmax': [], 'Vmn': [], 'Vstd': [],
            'OMEGAmin': [], 'OMEGAmax': [], 'OMEGAmn': [], 'OMEGAstd': [],
            'Hmin': [], 'Hmax': [], 'Hmn': [], 'Hstd': [],
        }

        self.stats = {
            'QVmn': 2.5e-3, 'QVstd': 3.805e-3,
            'Tmn': 254.738, 'Tstd': 29.034,
            'Umn': 1.922, 'Ustd': 12.837,
            'Vmn': 0.0528, 'Vstd': 7.071,
            'OMEGAmn': 2.461e-3, 'OMEGAstd': 0.1323,
            'Hmn': 11834.88, 'Hstd': 12125.891,
        }

        # filesystem locations
        # TODO: handle these for mswep/merra/am4 precip as target?
        self.source_dir = os.path.join(pth.SCRATCH, self.exp)
        self.target_dir = os.path.join(pth.SCRATCH, self.exp)

    def __len__(self):
        if self.weekly:
            # SPECIFYING FOR SPRING/SUMMER OR BOTH
            # length of dataset, total n_weeks in self.years for guaranteed 4 weeks per month
            #return 4 * self.n_months * len(self.years)
            return len(self.t_strs)
        else:  # assume monthly
            return self.n_months * len(self.years) # 12 * len(self.years)

    def __getitem__(self, idx):
        """
        select batch by idx -> month or week conversion
        """
        # TODO: presumably this is preconfigured to select for months/weeks/days etc.
        # idx passed in would reflect which time frame per batch
        t_str = self.t_strs[idx]
        source, target = self.get_source(t_str), self.get_target(t_str)
        if source.shape[0] != target.shape[0]:
            source, target = self.prune(source, target, t_str)
        sample = (source, target, t_str)
        return sample

    @staticmethod
    def prune(source, target, t_str):
        # if batch size doesn't match then trim to smaller
        print(f'pruning {t_str}...')
        trim = min(source.shape[0], target.shape[0])
        return source[:trim], target[:trim]

    @staticmethod
    def get_t_str(year, month, week=None):
        year, month = str(year), str(month + 1)
        t_str = year + month.zfill(2)
        t_str += f'_{week}' if week is not None else ''
        return t_str

    @staticmethod
    def regrid(ds, new_lat_step=0.5, new_lon_step=0.625):
        lat1, lat2 = round(ds.lat[0].item(), 2), round(ds.lat[-1].item(), 2)
        lon1, lon2 = round(ds.lon[0].item(), 2), round(ds.lon[-1].item(), 2)
        new_lat = np.arange(lat2, lat1, new_lat_step)
        new_lon = np.arange(lon1, lon2, new_lon_step)
        print(new_lat, new_lon)
        return ds.interp(lat=new_lat, lon=new_lon)

    @staticmethod
    def select_batch(ds, **kwargs):
        return ds.sel(**kwargs)

    def get_stats(self):
        for v in self.vars:
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

    def get_source(self, t_str):
        f = os.path.join(self.source_dir,  'Cfill.' + t_str + '.nc')
        # open_ is slightly faster than load_
        ds = xr.open_dataset(f)  # ...(f, chunks='auto') dask has issues here
        # ds = ds.sel(lev=self.p)  # to reduce number of plevels  included

        if self.rm_var is not None:
            ds = self._rm_var(ds)

        if STATS:
            for v in self.vars:
                dav = ds[v]
                self._stats[v + 'max'].append(dav.max().data)
                self._stats[v + 'min'].append(dav.min().data)
                self._stats[v + 'mn'].append(dav.mean().data)
                self._stats[v + 'std'].append(dav.std().data)
            return

        if NORM:
            for v in self.vars:
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']
        
        #da = ds.to_dataarray()
        source = torch.tensor(da.to_dataarray().data).permute(1, 0, 2, 3, 4)
        # if wanting to move away from 3D arch
        # s = source.shape
        # source = source.reshape(s[0], -1, s[3], s[4])
        
        # each chunk of source will be (time, var, (lev), lat, lon)
        return source

    def get_target(self, t_str):
        if ROI:
            # rough CUS crop
            lat_slc, lon_slc = slice(28, 52), slice(-108.125, -84.375)
        else:
            lat_slc, lon_slc = slice(None, None), slice(None, None)

        f = os.path.join(self.target_dir, 'P.' + t_str + '.nc')
        da = xr.open_dataset(f).sel(lat=lat_slc, lon=lon_slc).to_dataarray()
        target = torch.tensor(da.data).permute(1, 0, 2, 3)  # return as (time, 1, lat, lon)
        return target

## ================================================================================
if __name__ == '__main__':
    main()
