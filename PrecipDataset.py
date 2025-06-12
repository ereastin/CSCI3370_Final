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
ROI = True

# TODO: worth trying a version of this with open_mfdataset()?
# then could just grab it all at once and select what is needed at runtime
# how is mem handled then tho? is file kept open the whole time? probably.?
# if one week is selected is that array put in mem? how is it reused?

def main():
    t1 = time.time()
    pd = PrecipDataset('train', 'inc3d_reana', season='both', weekly=True)
    #s, t, tt = pd.__getitem__(20)
    #print(s.shape, t.shape)
    #return
    for i, (s, t, t_str) in enumerate(pd):
        print(s.shape, t.shape)
        exit()
        #if s.shape[0] != t.shape[0]:
        #    print(t_str, f'failed, wrong shapes: {s.shape}, {t.shape}')
    t2 = time.time()
    #pd.get_stats()  # TODO: was this ever run for random weekly stuff
    print(t2 - t1)

class PrecipDataset(Dataset):
    def __init__(self, mode, exp, season='both', weekly=False, rm_var=None, zero=False):
        super(PrecipDataset, self).__init__()

        """
        these according to ERA5 for ~CONUS
        driest: 2001, 2000, 2007, 2012, 1988  by increasing avg mm/day over the year e.g. 2001 is driest
        wettest: 1983, 2019, 2018, 1982, 1998 by decreasing avg mm/day over the year e.g. 1983 is wettest
        these were self picked... better way to select? is random better?
        """
        # these here are for original effort: train -> val -> test
        # self.years = [2001, 2002, 2004, 2005, 2006, 2008, 2009, 2010, 2012, 2014, 2015, 2016, 2017, 2019, 2020]
        # self.years = [2007, 2011, 2018]
        # self.years = [2003, 2013]
        self.season = season
        self.n_months = 6 if self.season == 'both' else 3
        yrs = list(map(str, range(1980, 2021)))
        mnth_offset = 6 if season == 'sum' else 3
        mnths = list(map(lambda x: str(x).zfill(2), range(mnth_offset, mnth_offset + self.n_months)))
        wks = list(map(str, range(4)))
        t_strs = np.asarray([f'{l[0]}{l[1]}_{l[2]}' for l in list(product(yrs, mnths, wks))])
        rng = np.random.default_rng(seed=4207765)  # fix the seed, at least one shuffle is the same everytime AFAIK..
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
        # these are for original version (Zhang)
        self.p = [1000., 850., 700., 500., 200., 100., 50., 10.]
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
        if self.weekly:
            #n_weeks = self.n_months * 4  # weeks per year
            #year = self.years[idx // n_weeks]
            #month_off = 5 if self.season == 'sum' else 2
            #week_of_yr = idx % n_weeks
            #month = month_off + (week_of_yr // 4)
            #week = week_of_yr - (week_of_yr // 4 * 4)
            t_str = self.t_strs[idx]
        else:
            month_off = 5 if self.season == 'sum' else 2
            year = self.years[idx // self.n_months]  # self.years[idx // 12]
            month = month_off + (idx % self.n_months)  # idx % 12

        # TODO: quick fix for model changes 
        #srcs, trgts = [], []
        #for i in range(4):
        #    t_str = self.get_t_str(year, month, week=i)
        #    srcs.append(self.get_source(t_str))
        #    trgts.append(self.get_target(t_str))
        #source = torch.cat(srcs, dim=0)
        #target = torch.cat(trgts, dim=0)
        # print(source.shape, target.shape)
        # t_str = self.get_t_str(year, month, week=week)
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

    def get_stats(self):
        for v in self.vars:
            vm, vs = np.mean(self._stats[v + 'mn']), np.mean(self._stats[v + 'std'])
            mx, mn = np.amax(self._stats[v + 'max']), np.amin(self._stats[v + 'min'])
            print(f'Variable {v}: mean {vm} std {vs} max {mx} min {mn}')

    def _rm_var(self, ds):
        # set variable to average value at each pressure-height.?
        lev = ds.coords['lev']
        for p in self.p:
            if self.zero:
                fill = 0.0
            else:
                fill = ds[self.rm_var].sel(dict(lev=lev[lev == p])).mean()
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
        
        da = ds.to_dataarray()
        if DEEP:
            source = torch.tensor(da.data).permute(1, 0, 2, 3, 4)
            # if wanting to move away from 3d arch
            # s = source.shape
            # source = source.reshape(s[0], -1, s[3], s[4])
        else:
            times = da['time'].data
            source = [torch.tensor(da.sel(time=str(t)).data).reshape(-1, da.shape[3], da.shape[4]).unsqueeze(0) for t in times]
            source = torch.cat(source, dim=0)
        
        # each chunk of source will be (time, lev, lat, lon)
        return source

    def get_target(self, t_str):
        if ROI:
            # too small i think to be valuable
            lat_slc, lon_slc = slice(24, 50.5), slice(-108.75, -83.125)
        else:
            lat_slc, lon_slc = slice(None, None), slice(None, None)

        f = os.path.join(self.target_dir, 'P.' + t_str + '.nc')
        da = xr.open_dataset(f).sel(lat=lat_slc, lon=lon_slc).to_dataarray()
        target = torch.tensor(da.data).permute(1, 0, 2, 3)  # return as (time, 1, lat, lon)
        return target


if __name__ == '__main__':
    main()
