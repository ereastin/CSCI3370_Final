# -*- coding: utf-8 -*-
"""PrecipDataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YIIIIXUIOXusBABYt_AsjaK2hupZhYKV
"""

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import os
from cdo import *
import time

class PrecipDataset(Dataset):
    def __init__(self, mode):
        """
        ERA5 filenames: {var}.{YYYY}{MM}.nc
        MSWEP filenames: {YYYY}{DDD}.{HH}.nc, DDD = [001...366], HH = [00, 03, 06, 09, 12, 15, 18, 21], assuming includes leap years..?
          -> if dealing with leap years then ensure both ERA and MSWEP have the same data somehow

        ERA data (4x daily data at 1.5 degree grid) Zhang et al. use 0.25 deg.? need to upsample?

        data @ 38 pressure levels (potentially 304 channels)
          potential vorticity (PV)
          relative vorticity (VO)
          temperature (T)
          specific humidity (Q)  ... missing Q data for 80s 90s.? others? just use 2000+ then?
          zonal winds (U)
          meridional winds (V)
          vertical winds (W)
          geopotential (Z)

        surface (??) data (potentially 6 additional layers)
          U, V, Q, Z, T, P (this SLP?)

        MSWEP data (8x daily data at .1 degree grid)
        """
        super(PrecipDataset, self).__init__()
        if mode == 'train':
            self.years = [2001, 2002, 2004, 2005, 2006, 2008, 2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2019, 2020]
        elif mode == 'val':
            self.years = [2007]
        elif mode == 'test':
            self.years = [2003, 2013, 2018]

        # num days per month by index, adjust for leap year in func
        self.hrs = [('00', '03'), ('06', '09'), ('12', '15'), ('18', '21')]
        self.dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.vars = ['Q', 'T', 'VO', 'PV', 'U', 'V', 'W', 'Z']
        self.p = ['1000', '850', '700', '500', '200', '100', '50', '10']  # mbar, what pressure heights do we want.?

        # filesystem locations
        self.target_dir = '/projects/bccg/andromeda1/bccg/Shared_Data/OBS/MSWEP/Past/3hourly/'
        self.source_dir = '/projects/bccg/andromeda1/bccg/Shared_Data/REANA/ERA5/4xdaily/'
        self.scratch_dir = '/scratch/eastinev/'

        self.cdo = Cdo(tempdir=self.scratch_dir)
        self.cdo.debug = True
        self.cdo.cleanTempDir()  # make sure to do this after completing, leave here for debugging now
        
        # remapping utils
        self.era_wts_gen = False
        self.mswep_wts_gen = False
        self.mswep_remap_wts = os.path.join(self.scratch_dir, 'mswep_weights.nc')
        self.era_remap_wts = os.path.join(self.scratch_dir, 'era_weights.nc')
        self.mswep_gfile = os.path.join(self.scratch_dir, 'mswep_gridfile.txt')
        self.era_gfile = os.path.join(self.scratch_dir, 'era_gridfile.txt')
        # these should take care of cropping to region and up/down sampling
        self.mswep_gfile_text = [
            'gridtype  = lonlat',
            'gridsize  = 9216',
            'xsize     = 96',
            'ysize     = 96',
            'xname     = lon',
            'xlongname = "longitude"',
            'xunits    = "degrees_east"',
            'yname     = lat',
            'ylongname = "latitude"',
            'yunits    = "degrees_north"',
            'xfirst    = -108.0',
            'xinc      = 0.25',
            'yfirst    = 24.0',
            'yinc      = 0.25'
        ]
        self.era_gfile_text = [
            'gridtype  = lonlat',
            'gridsize  = 9216',
            'xsize     = 96',
            'ysize     = 96',
            'xname     = lon',
            'xlongname = "longitude"',
            'xunits    = "degrees_east"',
            'yname     = lat',
            'ylongname = "latitude"',
            'yunits    = "degrees_north"',
            'xfirst    = 252.0',
            'xinc      = 0.25',
            'yfirst    = 24.0',
            'yinc      = 0.25'
        ]
        self.write_gridfiles()

    def __len__(self):
        # length of dataset, total n_months in self.years
        return 12 * len(self.years)

    def __getitem__(self, idx):
        """
        select batch by idx -> month conversion
        each month is a batch -> a month is minimum necessary condition statistically (?)
        should then be able to shuffle months but still preserve temporal associations
        """
        assert type(idx) is int

        year = self.years[idx // 12]
        month = int((idx / 12 - idx // 12) * 12)  # 0-indexed, Jan == 0, Feb == 1, etc.

        era_t = self.get_era_t(year, month)
        mswep_t = self.get_mswep_t(year, month)
        #source = self.get_source(era_t)
        target = self.get_target(year, month, mswep_t)

        #sample = {'source': source, 'target': target}
        return #sample

    def write_gridfiles(self):
        if not os.path.exists(self.era_gfile):
            with open(self.era_gfile, 'w+') as f:
                f.write('\n'.join(self.era_gfile_text))

        if not os.path.exists(self.mswep_gfile):
            with open(self.mswep_gfile, 'w+') as f:
                f.write('\n'.join(self.mswep_gfile_text))

    def write_remap_wts(self, gridfile, infile, wt_file):  # or if we want to select pressure levels before remap just trash this
        if not os.path.exists(wt_file):  # cna probably just move this check into the compose* funcs, dont need *wts_gen flag
            self.cdo.genbil(gridfile, input=infile, output=wt_file)

    def get_era_t(self, year, month):
        year, month = str(year), str(month + 1)
        return year + month.zfill(2)

    def get_mswep_t(self, year, month):
        leap_year = (year % 4 == 0)
        days = self.dpm[month]
        days += 1 if leap_year and month == 1 else 0
        return torch.arange(1, days + 1).tolist()

    def get_source(self, t_str):
        """
        ERA5 indexed as 'lat', 'lon', 'time', 'lev'
        """
        source = []
        for var in self.vars:
            fname = var + '.' + t_str + '.nc'  # {var}.{YYYY}{MM}.nc
            print(fname)
            var_pth = os.path.join(self.source_dir, var, fname)
            if not self.era_wts_gen:
                t_gen1 = time.time()
                self.cdo.genbil(self.era_gfile, input=f'-sellevel,{",".join(self.p)} {var_pth}', output=self.era_remap_wts)
                t_gen2 = time.time()
                # self.write_remap_wts(self.era_gfile, var_pth, self.era_remap_wts)
                self.era_wts_gen = True
                print(f'Map weights generation time: {t_gen2 - t_gen1}')

            # do remapping -> does this handle/preserve time and pressure dims..?
            outfile = os.path.join(self.scratch_dir, f'{var}.nc')
            t_map1 = time.time()
            self.cdo.remap(self.era_gfile, self.era_remap_wts, input=f'-sellevel,{",".join(self.p)} {var_pth}', output=outfile)
            t_map2 = time.time()
            print(f'Remap time: {t_map2 - t_map1}')
            ds = xr.open_dataset(outfile)
            source.append(torch.tensor(ds.to_dataarray().data).squeeze(0))
            # TODO: make tensors of the times and try the same reshaping? to verify below works?

        # each chunk of source will be (time, lev, lat, lon)
        source = torch.stack(source, dim=1)  # (time, N, lev, lat, lon) stack along new dim N vars
        source = source.reshape(source.shape[0], -1, source.shape[3], source.shape[4])  # voodoo dim magic -> (time, N * lev, lat, lon)
        return source

    def get_target(self, year, month, day_rng):
        t1 = time.time()
        # do merge and remap
        t_strt = '00:00:00'  # correct?
        d_strt = f'{str(year)}-{str(month + 1).zfill(2)}-01'  # this should always start as YYYY-MM-01
        outfile = os.path.join(self.scratch_dir, 'target.nc')
        day_rng = [1]
        fnames = [f'{str(year)}{str(sum(self.dpm[:month]) + day).zfill(3)}*' for day in day_rng]
        files = [os.path.join(self.target_dir, f) for f in fnames] 
        files = ' '.join(files)
        if not self.mswep_wts_gen:
            #self.write_remap_wts(self.mswep_gfile, files[0][:-1] + '.00.nc', self.mswep_remap_wts)
            #if not os.path.exists(self.mswep_remap_wts):
            self.cdo.gencon(self.mswep_gfile, input=f'-inttime,{d_strt},{t_strt},6hour -mergetime -apply,-sellonlatbox,-110,-80,20,55 [ {files} ]', output=self.mswep_remap_wts)
            self.mswep_wts_gen = True

        # this remapping is wayy slower like 5 mins vs 2 secs for one month... not sure there's anything to be done about it think its just slow going this way..?
        self.cdo.remap(self.mswep_gfile, self.mswep_remap_wts, input=f'-inttime,{d_strt},{t_strt},6hour -mergetime -apply,-sellonlatbox,-110,-80,20,55 [ {files} ]', output=outfile)
        target = torch.tensor(xr.load_dataset(outfile).to_dataarray().data).permute(1, 0, 2, 3)
        t2 = time.time()
        print(f'compose time: {(t2 - t1) / len(day_rng)}')
        return target
