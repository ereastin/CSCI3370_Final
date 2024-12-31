import os
import sys
from cdo import *
import xarray as xr
import torch

def main():
    year, month = int(sys.argv[1]), int(sys.argv[2])
    mm = MakeMonthly()
    try:
        # preprocess mswep data
        day_rng = mm.get_mswep_t(year, month)
        mm.get_target(year, month, day_rng)

        # preprocess era data
        t_str = mm.get_t_str(year, month)
        mm.get_source(t_str)
    except Exception as e:
        print(f'{year}-{month} Failed with exception: {e}')
        mm.clean()

    print(f'{year}-{month} Completed')
    mm.clean()

# TODO: this doesn't really need to be a class
class MakeMonthly:
    def __init__(self):
        self.source_dir = '/projects/bccg/andromeda1/bccg/Shared_Data/REANA/ERA5/4xdaily/'
        self.target_dir = '/projects/bccg/andromeda1/bccg/Shared_Data/OBS/MSWEP/Past/3hourly/'
        self.scratch_dir = '/scratch/eastinev/'

        self.cdo = Cdo(tempdir=self.scratch_dir)
        # self.cdo.debug = True

        self.mswep_wts =  os.path.join(self.scratch_dir, 'mswep_weights.nc') 
        self.era_wts = os.path.join(self.scratch_dir, 'era_weights.nc')
        self.dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.vars = ['Q', 'T', 'U', 'V', 'Z']  # add in topography, remove Q? train without Q or train with and remove for test?
        # 2PVU data is variables at the 2 potential vorticity units "surface"
        # this is a crude definition of the troposphere-stratosphere boundary..
        # if removing Q can probably take ERA5 data from farther back, that only thing missing?
        # re: pressure levels -> 250 mb ~top of troposphere, remove from strat and add more lower?
        self.p = ','.join(['1000', '950', '850', '700', '500', '350', '200', '100'])  # mbar, what pressure heights do we want.?
        self.bbox = ','.join(['-142,-48,12,56'])
        self.mswep_gfile =  os.path.join(self.scratch_dir, 'mswep_gridfile.txt')
        self.era_gfile = os.path.join(self.scratch_dir, 'era_gridfile.txt')
        self.mswep_gfile_txt = '\n'.join([
            'gridtype  = lonlat',
            # 'gridsize  = 9216',
            # 'xsize     = 96',
            # 'ysize     = 96',
            # CONUS sizes
            'gridsize  = 57600',
            'xsize     = 360',
            'ysize     = 160',
            'xname     = lon',
            'xlongname = "longitude"',
            'xunits    = "degrees_east"',
            'yname     = lat',
            'ylongname = "latitude"',
            'yunits    = "degrees_north"',
            #'xfirst    = -108.0',
            'xfirst    = -140.0',
            'xinc      = 0.25',
            #'yfirst    = 24.0',
            'yfirst    = 13.0',
            'yinc      = 0.25'
        ])
        self.era_gfile_text = '\n'.join([
            'gridtype  = lonlat',
            # 'gridsize  = 9216',
            # 'xsize     = 96',
            # 'ysize     = 96',
            # CONUS sizes
            'gridsize  = 57600',
            'xsize     = 360',
            'ysize     = 160',
            'xname     = lon',
            'xlongname = "longitude"',
            'xunits    = "degrees_east"',
            'yname     = lat',
            'ylongname = "latitude"',
            'yunits    = "degrees_north"',
            #'xfirst    = 252.0',
            'xfirst    = 220.0',
            'xinc      = 0.25',
            #'yfirst    = 24.0',
            'yfirst    = 13.0',
            'yinc      = 0.25'
        ])

    def prep(self):
        self.write_gfiles()
        self.gen_wts()
        self.get_topo()

    def get_topo(self):
        f = os.path.join(self.scratch_dir, 'global_EL.nc4')
        of = os.path.join(self.scratch_dir, 'region_EL.nc4')
        self.cdo.sellonlatbox(
            input=f'-140.0,-49.75,13.0,54.75 {f}',
            output=of
        )

    def write_gfiles(self):
        with open(self.era_gfile, 'w+') as f:
            f.write(self.era_gfile_txt)

        with open(self.msewp_gfile, 'w+') as f:
            f.write(self.mswep_gfile_txt)

    def get_t_str(self, year, month):
        year, month = str(year), str(month + 1)
        return year + month.zfill(2)

    def get_mswep_t(self, year, month):
        leap_year = (year % 4 == 0)
        days = self.dpm[month]
        days += 1 if leap_year and month == 1 else 0
        d0 = 2 if leap_year and month > 1 else 1
        return list(range(d0, days + d0))

    def get_source_files(self, t_str):
        atm_files = [os.path.join(self.source_dir, f'{var}', f'{var}.{t_str}.nc') for var in self.vars]
        atm_files = ' '.join(atm_files)
        outfile = os.path.join(self.scratch_dir, 'era5', f'C.{t_str}.nc')
        return atm_files,  outfile

    def get_source(self, t_str):
        atm_files, outfile = self.get_source_files(t_str)
        of_atm = os.path.join(self.scratch_dir, 'tmp_atm.nc')
        of_topo = os.path.join(self.scratch_dir, 'region_EL.nc4')

        self.cdo.remap(
            self.era_gfile,
            self.era_wts,
            input=f'-merge -apply,"-sellevel,{self.p} -sellonlatbox,{self.bbox}" [ {atm_files} ]',
            output=of_atm
        )
        self.cdo.merge(
            input=f'{of_atm} {of_topo}',
            output=outfile
        )

    def get_target_files(self, year, month, day_rng):
        outfile = os.path.join(self.scratch_dir, 'mswep', f'P.{str(year)}{str(month + 1).zfill(2)}.nc')
        fnames = [f'{str(year)}{str(sum(self.dpm[:month]) + day).zfill(3)}*' for day in day_rng]
        files = [os.path.join(self.target_dir, f) for f in fnames]  
        files = ' '.join(files)
        return files, outfile

    def get_target(self, year, month, day_rng):
        t_strt = '00:00:00'  # combining month will always start at 00:00:00
        d_strt = f'{str(year)}-{str(month + 1).zfill(2)}-01'  # this should always start as YYYY-MM-01
        files, outfile = self.get_target_files(year, month, day_rng)

        self.cdo.remap(
            self.mswep_gfile,
            self.mswep_wts,
            input=f'-inttime,{d_strt},{t_strt},6hour -mergetime -apply,-sellonlatbox,{self.bbox} [ {files} ]',
            output=outfile
        )

    def gen_wts(self):
        f = os.path.join(self.source_dir, 'T.200101.nc')
        self.cdo.genbil(
            self.era_gfile,
            input=f'-sellevel,{self.p} -sellonlatbox,{self.bbox} {f}',
            output=self.era_wts
        )

        f = os.path.join(self.target_dir, '2001001.00.nc')
        self.cdo.gencon(
            self.mswep_gfile,
            input=f'-sellonlatbox,{self.bbox} {f}',
            output=self.mswep_wts
        )

    def clean(self):
        self.cdo.cleanTempDir()

if __name__ == '__main__':
    main()

