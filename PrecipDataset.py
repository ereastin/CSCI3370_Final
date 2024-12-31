import torch
from torch.utils.data import Dataset
import xarray as xr
import os
import time

class PrecipDataset(Dataset):
    def __init__(self, mode, rm_var=None):
        """
        ERA5 filenames: {var}.{YYYY}{MM}.nc
        MSWEP filenames: {YYYY}{DDD}.{HH}.nc, DDD = [001...366], HH = [00, 03, 06, 09, 12, 15, 18, 21]

        ERA data (4x daily data at 1.5 degree grid)

        data @ 38 pressure levels (potentially 304 channels)
          potential vorticity (PV)
          relative vorticity (VO)
          temperature (T)
          specific humidity (Q)
          zonal winds (U)
          meridional winds (V)
          vertical winds (W)
          geopotential (Z)

        MSWEP data (8x daily data at .1 degree grid)
        """
        super(PrecipDataset, self).__init__()
        if mode == 'train':
            self.years = [2001, 2002, 2004, 2005, 2006, 2008, 2009, 2010, 2012, 2014, 2015, 2016, 2017, 2019, 2020]
        elif mode == 'val':
            self.years = [2007, 2011, 2018]
        elif mode == 'test':
            self.years = [2003, 2013]

        self.rm_var = rm_var

        # filesystem locations
        self.scratch_dir = '/scratch/eastinev/'
        self.source_dir = os.path.join(self.scratch_dir, 'era5')
        self.target_dir = os.path.join(self.scratch_dir, 'mswep')

    def __len__(self):
        # length of dataset, total n_months in self.years
        return 12 * len(self.years)

    def __getitem__(self, idx):
        """
        select batch by idx -> month conversion
        each month is a batch -> a month is minimum necessary condition statistically (?)
        should then be able to shuffle months but still preserve temporal associations
        """
        year = self.years[idx // 12]
        month = idx % 12
        t_str = self.get_t_str(year, month)
        source = self.get_source(t_str)
        target = self.get_target(t_str)
        sample = (source, target, t_str)
        return sample

    def get_t_str(self, year, month):
        year, month = str(year), str(month + 1)
        return year + month.zfill(2)

    def get_source(self, t_str):
        """
        ERA5 indexed as 'lat', 'lon', 'time', 'lev'
        Data shape options for including elevation:
            5D volumetric input (B, C, H, W, D)
                - add an extra channel on all but only include elevation at lowest, 0s else?
                - add an elevation 'layer' that batches pass through as a boundary condition? how would this work?
                - do we even need to care? ERA5 adjusted for elevation.?
                - start here without elevation?
            4D channel-only input (B, C, H, W)
                - easily include elevation as an extra channel
        """
        f = os.path.join(self.source_dir,  'C.' + t_str + '.nc')
        ds = xr.load_dataset(f)
        if self.rm_var is not None:
            # fill = 273
            # TODO: best way to handle temperature? others? enforce average everywhere?
            # if not Q then wtf did it learn?
            ds[self.rm_var] = xr.full_like(ds[self.rm_var], fill_value=0.0)
        
        da = ds.to_dataarray()
        times = da['time'].data
        # straight reshaping of data array not equivalent.. wonder if there is a way tho this is ~ 4x slower
        # are we sure this is right.? should it even matter?
        source = [torch.tensor(da.sel(time=str(t)).data).reshape(-1, da.shape[3], da.shape[4]).unsqueeze(0) for t in times]
        source = torch.cat(source, dim=0)
        # each chunk of source will be (time, lev, lat, lon)
        return source

    def get_target(self, t_str):
        f = os.path.join(self.target_dir, 'P.' + t_str + '.nc')
        target = torch.tensor(xr.load_dataset(f).to_dataarray().data).permute(1, 0, 2, 3)  # return as (time, 1, lat, lon)
        return target

