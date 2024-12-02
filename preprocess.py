import os
import sys
from cdo import *
import xarray as xr
import torch

def main():
    year, month = int(sys.argv[1]), int(sys.argv[2])
    mm = MakeMonthly()
    try:
        #day_rng = mm.get_mswep_t(year, month)
        #mm.get_target(year, month, day_rng)
        t_str = mm.get_t_str(year, month)
        mm.get_source(t_str)
    except Exception as e:
        print(f'{year}-{month} Failed with exception: {e}')
        mm.clean()

    print(f'{year}-{month} Completed')
    mm.clean()

class MakeMonthly:
    def __init__(self):
        #self.wts_gen = False
        self.source_dir = '/projects/bccg/andromeda1/bccg/Shared_Data/REANA/ERA5/4xdaily/'
        self.target_dir = '/projects/bccg/andromeda1/bccg/Shared_Data/OBS/MSWEP/Past/3hourly/'
        self.scratch_dir = '/scratch/eastinev/'

        self.cdo = Cdo(tempdir=self.scratch_dir)
        # self.cdo.debug = True

        self.mswep_wts =  os.path.join(self.scratch_dir, 'mswep_weights.nc') 
        self.era_wts = os.path.join(self.scratch_dir, 'era_weights.nc')
        self.dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.vars = ['Q', 'T', 'VO', 'PV', 'U', 'V', 'W', 'Z']
        self.p = ['1000', '850', '700', '500', '200', '100', '50', '10']  # mbar, what pressure heights do we want.?
        self.mswep_gfile =  os.path.join(self.scratch_dir, 'mswep_gridfile.txt')
        self.era_gfile = os.path.join(self.scratch_dir, 'era_gridfile.txt')
        self.mswep_gfile_txt = [
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
        self.write_gfiles()

    def write_gfiles(self):
        if not os.path.exists(self.era_gfile):
            with open(self.gfile, 'w+') as f:
                f.write('\n'.join(self.era_gfile_txt))
        
        if not os.path.exists(self.mswep_gfile):
            with open(self.gfile, 'w+') as f:
                f.write('\n'.join(self.mswep_gfile_txt))

    def get_t_str(self, year, month):
        year, month = str(year), str(month + 1)
        return year + month.zfill(2)

    def get_mswep_t(self, year, month):
        leap_year = (year % 4 == 0)
        days = self.dpm[month]
        days += 1 if leap_year and month == 1 else 0
        d0 = 2 if leap_year and month > 1 else 1
        return list(range(d0, days + d0))

    def get_source(self, t_str):
        files = [os.path.join(self.source_dir, f'{var}', f'{var}.{t_str}.nc') for var in self.vars]
        files = ' '.join(files)
        outfile = os.path.join(self.scratch_dir, 'era5', f'C.{t_str}.nc')

        # remember to use cdo.gencon/genbil if weights haven't been created!
        self.cdo.remap(
            self.era_gfile,
            self.era_wts,
            input=f'-merge -apply,-sellevel,{",".join(self.p)} [ {files} ]',
            output=outfile
        )
        return

    def get_target(self, year, month, day_rng):
        t_strt = '00:00:00'  # combining month will always start at 00:00:00
        d_strt = f'{str(year)}-{str(month + 1).zfill(2)}-01'  # this should always start as YYYY-MM-01
        outfile = os.path.join(self.scratch_dir, 'mswep', f'P.{str(year)}{str(month + 1).zfill(2)}.nc')
        fnames = [f'{str(year)}{str(sum(self.dpm[:month]) + day).zfill(3)}*' for day in day_rng]
        files = [os.path.join(self.target_dir, f) for f in fnames]  
        files = ' '.join(files)

        # remember to use cdo.gencon/genbil if weights haven't been created yet
        self.cdo.remap(
            self.mswep_gfile,
            self.mswep_wts,
            input=f'-inttime,{d_strt},{t_strt},6hour -mergetime -apply,-sellonlatbox,-110,-80,20,55 [ {files} ]',
            output=outfile
        )
        return

    def clean(self):
        self.cdo.cleanTempDir()

if __name__ == '__main__':
    main()

