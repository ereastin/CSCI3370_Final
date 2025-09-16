import os
import sys
from cdo import *
import xarray as xr
from dask import array
import numpy as np
import glob
from argparse import ArgumentParser
import concurrent.futures as cf
sys.path.append('/home/eastinev/AI')
import paths as pth
import time
#from dask.distributed import Client

def main():
    t1 = time.time()
    year, month, prep, exp, weekly, reproc = parse_args()
    #xstr = f'{year}{str(month+1).zfill(2)}_'
    #curr_files = os.listdir(os.path.join(pth.SCRATCH, exp))
    #if 'Cfill.' + xstr + '3.nc' in curr_files and 'P.' + xstr + '3.nc' in curr_files: 
    #    print(f'{month + 1}, {year} already complete')
    #    return
    """
    else:
        mcf = glob.glob(os.path.join(pth.SCRATCH, exp, 'MC.' + xstr + '*'))
        pf = glob.glob(os.path.join(pth.SCRATCH, exp, 'P.' + xstr + '*'))
        for m in mcf:
            os.remove(m)
        for p in pf:
            os.remove(p)
        print(f'Processing {month + 1}, {year}')
    """

    mm = MakeMonthly(exp, weekly)
    if reproc:
        mm.reprocess(year, month)
    elif prep:
        mm.prep()
    else:
        try:
            # preprocess merra
            #mm.get_merra(year, month)
            # preprocess mswep data
            mm.get_mswep(year, month, six_hourly=False)
            # preprocess era data
            #mm.get_era(year, month)
        except Exception as e:
            print(f'{year}-{month} Failed with exception: {e}')
            mm.clean()

        print(f'{month + 1}, {year} completed')
    
    t2 = time.time()
    print(t2 - t1)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('year', type=int)
    parser.add_argument('month', type=int)
    parser.add_argument('-p', '--prep', action='store_true')
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-w', '--weekly', action='store_true')
    parser.add_argument('-r', '--reproc', action='store_true')
    args = parser.parse_args()
    return args.year, args.month, args.prep, args.exp, args.weekly, args.reproc

class MakeMonthly:
    def __init__(self, exp=None, weekly=False):
        """
        Updating for MERRA2/CAM6(?) data:
            1980-2022 at 3 or 6-hourly intervals (which?), spring/summer only
        """
        self.exp = exp
        self.weekly = weekly
        self.cdo = Cdo(tempdir=pth.SCRATCH)
        #self.cdo.debug = True

        self.dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        # TODO: my god this is ugly... better way?
        self.splits = {
            31: [slice(0, 8), slice(8, 16), slice(16, 24), slice(24, 31)], #[slice(0, 64), slice(64, 128), slice(128, 192), slice(192, 248)],
            30: [slice(0, 8), slice(8, 16), slice(16, 23), slice(23, 30)], #[slice(0, 64), slice(64, 128), slice(128, 184), slice(184, 240)],
            29: [slice(0, 8), slice(8, 15), slice(15, 22), slice(22, 29)], #[slice(0, 64), slice(64, 120), slice(120, 176), slice(176, 232)],
            28: [slice(0, 7), slice(7, 14), slice(14, 21), slice(21, 28)], #[slice(0, 56), slice(56, 112), slice(112, 168), slice(168, 224)]
        }

        self.era_vars = ['Q', 'T', 'U', 'V', 'Z', 'W', 'VO', 'PV']  # these are original 8
        self.merra_vars = ','.join(['U', 'V', 'OMEGA', 'H', 'T', 'QV'])  # if interested could use cloud ice and liquid mass mixing ratios?
        self.p = ','.join(np.array([
            1000, 975, 950, 925, 900, 875, 850,
            825, 800, 775, 750, 725, 700, 650,
            600, 550, 500, 450, 400, 350, 300,
            250, 200, 150, 100, 70, 50, 40,
            30, 20, 10, 7, 5, 4, 3
        ], dtype=str))
        # pressure levels used in original work
        # self.p = ','.join(['1000', '950', '850', '700', '500', '350', '200', '100'])
        self.bbox = '-140,-50.625,14,53.5'
        self.mswep_bbox = '-98.125,-89.375,31,52'
        self.merra_wts = os.path.join(pth.SCRATCH, f'{exp}', f'merra_weights.nc')  # DONT NEED THIS NOW!
        self.mswep_wts =  os.path.join(pth.SCRATCH, f'{exp}', f'mswep_weights.nc') 
        self.era_wts = os.path.join(pth.SCRATCH, f'{exp}', f'era_weights.nc')
        self.gfile =  os.path.join(pth.SCRATCH, f'{exp}', f'gridfile.txt')
        self.gfile_txt = '\n'.join([
            'gridtype  = lonlat',
            'gridsize  = 11520',
            'xsize     = 144',
            'ysize     = 80',
            'xname     = lon',
            'xlongname = "longitude"',
            'xunits    = "degrees_east"',
            'yname     = lat',
            'ylongname = "latitude"',
            'yunits    = "degrees_north"',
            'xfirst    = -140.0',
            'xinc      = 0.625',
            'yfirst    = 14.0',
            'yinc      = 0.5'
        ])

    def prep(self):
        self.write_gfile()
        self.gen_wts()
        #self.get_topo()

    def get_topo(self):
        # if including need to coarsen to 0.5x0.625 deg grid
        of = os.path.join(pth.SCRATCH, 'region_EL.nc')
        self.cdo.sellonlatbox(  # TODO: verify these are right or just use for all
            input=f'-140.0,-49.75,13.0,54.75 {pth.TOPO_2p5}',
            output=of
        )

    def write_gfile(self):
        with open(self.gfile, 'w+') as f:
            f.write(self.gfile_txt)

    def reprocess(self, year, month):
        fbase = f'{year}{str(month + 1).zfill(2)}'
        
        existing = os.listdir(os.path.join(pth.SCRATCH, self.exp))
        fnames = [os.path.join(pth.SCRATCH, self.exp, f'Cfill.{fbase}_{i}.nc') for i in range(4)]
        fs = ' '.join(fnames)        
        inf = os.path.join(pth.SCRATCH, self.exp, f'C.{fbase}.nc')
        if inf.split('/')[-1] in existing:
            ds = xr.open_dataset(inf, chunks='auto')
            ndays = len(ds.time) // 8
            splits = self.splits[ndays]
            for i, f in enumerate(fnames):
                t_slc = splits[i]
                ds.isel(time=t_slc).to_netcdf(f)
            
            os.remove(inf)
        
        fnames = [os.path.join(pth.SCRATCH, self.exp, f'P.{fbase}_{i}.nc') for i in range(4)]
        fs = ' '.join(fnames)        
        inf = os.path.join(pth.SCRATCH, self.exp, f'P.{fbase}.nc')
        if inf.split('/')[-1] in existing:
            ds = xr.open_dataset(inf, chunks='auto')
            ndays = len(ds.time) // 8
            splits = self.splits[ndays]
            for i, f in enumerate(fnames):
                t_slc = splits[i]
                ds.isel(time=t_slc).to_netcdf(f)
            
            os.remove(inf)

    def get_merra_t_str(self, year, month):
        # fname: MERRA2_N00.inst3_3d_asm_Np.YYYYMMDD.nc4
        # N is 1-4 (?) 1 for 1980-1991, 2 for 1992-2000, 3 for 2001-2010, 4 for 2011+
        leap_year = (year % 4 == 0)
        days = self.dpm[month]
        days += 1 if leap_year and month == 1 else 0
        return [str(year) + str(month + 1).zfill(2) + str(day).zfill(2) for day in range(1, days + 1)]

    def get_merra_files(self, year, month):
        t_strs = self.get_merra_t_str(year, month)
        infiles = [os.path.join(pth.MERRA, str(year), f'*{t_str}.nc4') for t_str in t_strs]
        splits = self.splits[len(t_strs)] if self.weekly else [slice(None, None)]
        infiles = [' '.join(infiles[slc]) for slc in splits]
        outfile = os.path.join(pth.SCRATCH, f'{self.exp}', f'MC.{year}{str(month + 1).zfill(2)}')
        return infiles, outfile

    def combine_merra(self, infiles, outfile):
        self.cdo.mergetime(
            input=f'[ -selname,{self.merra_vars} -sellonlatbox,{self.bbox} : {infiles} ]',
            output=outfile
        )

    def get_merra(self, year, month):
        infiles, outfile = self.get_merra_files(year, month)
        outfiles = [outfile + f'_{i}.nc' for i in range(len(infiles))]
        # TODO: as with mswep stuff, if moving to 6-hourly will need to handle inttime stuff with this
        for infs, outf in zip(infiles, outfiles):
            self.combine_merra(infs, outf)

        # could not get this to work with sbatch..
        # with cf.ThreadPoolExecutor(max_workers=4) as ex:
        #     ex.map(self.combine_merra, infiles, outfiles)

    def get_era_t_str(self, year, month):
        # fname: {VAR}.YYYYMM.nc
        year, month = str(year), str(month + 1)
        return year + month.zfill(2)

    def get_era_files(self, t_str):
        infiles = [os.path.join(pth.ERA5_4xd, f'{var}', f'{var}.{t_str}.nc') for var in self.era_vars]
        infiles = ' '.join(infiles)
        outfile = os.path.join(pth.SCRATCH, f'{self.exp}', f'EC.{t_str}.nc')
        return infiles,  outfile

    def get_era(self, year, month):
        t_str = self.get_era_t_str(year, month)
        infiles, outfile = self.get_era_files(t_str)

        self.cdo.remap(
            self.gfile,
            self.era_wts,
            input=f'-merge [ -sellevel,{self.p} -sellonlatbox,{self.bbox} : {infiles} ]',
            output=outfile
        )

    def get_mswep_t_str(self, year, month):
        # fname: YYYYDDD.HH
        leap_year = (year % 4 == 0)
        days = self.dpm[month]
        days += 1 if leap_year and month == 1 else 0
        d0 = 2 if leap_year and month > 1 else 1
        return [str(year) + str(sum(self.dpm[:month]) + day).zfill(3) for day in range(d0, days + d0)]

    def get_mswep_files(self, year, month):
        t_strs = self.get_mswep_t_str(year, month)
        infiles = [os.path.join(pth.MSWEP, f'{t_str}*') for t_str in t_strs]
        splits = self.splits[len(t_strs)] if self.weekly else [slice(None, None)]
        infiles = [' '.join(infiles[slc]) for slc in splits]
        outfile = os.path.join(pth.SCRATCH, f'{self.exp}', f'P.{year}{str(month + 1).zfill(2)}')
        return infiles, outfile

    def combine_mswep(self, infiles, outfile):
        self.cdo.remap(
            self.gfile,
            self.mswep_wts,
            input=f'-mergetime [ -sellonlatbox,{self.bbox} : {infiles} ]',
            output=outfile
        )

    def get_mswep(self, year, month, six_hourly=False):
        infiles, outfile = self.get_mswep_files(year, month)
        if six_hourly:
            # TODO: if moving to 6-hourly and weekly chunks this needs an update
            t_strt = '00:00:00'  # combining month will always start at 00:00:00
            d_strt = f'{str(year)}-{str(month + 1).zfill(2)}-01'  # this should always start as YYYY-MM-01
            self.cdo.remap(
                self.gfile,
                self.mswep_wts,
                input=f'-inttime,{d_strt},{t_strt},6hour -mergetime -apply,-sellonlatbox,{self.bbox} [ {infiles} ]',
                output=outfile
            )
        else:
            outfiles = [outfile + f'_{i}.nc' for i in range(len(infiles))]
            for infs, outf in zip(infiles, outfiles):
                self.combine_mswep(infs, outf)

            # could not get hyperthreading to work with sbatch, worked on interactive tho..
            # with cf.ThreadPoolExecutor(max_workers=4) as ex:
            #     ex.map(self.combine_mswep, infiles, outfiles)

    def gen_wts(self):
        # infiles, _ = self.get_era_files(2001, 1)
        # self.cdo.genbil(
        #     self.gfile,
        #     input=f'-merge -apply,"-sellevel,{self.p} -sellonlatbox,{self.bbox}" [ {infiles} ]',
        #     output=self.era_wts
        # )
        
        # infiles, _ = self.get_merra_files(2001, 1)
        # self.cdo.genbil(
        #     self.gfile,
        #     input=f'-merge -apply,"-selname,{self.merra_vars} -sellonlatbox,{self.bbox}" [ {infiles} ]',
        #     output=self.merra_wts
        # )

        infiles, _ = self.get_mswep_files(2001, 1)
        f = infiles[0].split(' ')[0][:-1] + '.00.nc'
        # TODO: handle this for 6hourly or not
        # also how exactly does remap handle 3d fields... can i get weights for one level and apply to all?
        self.cdo.gencon(
            self.gfile,
            # input=f'-inttime,{d_strt},{t_strt},6hour -mergetime -apply,-sellonlatbox,{self.bbox} [ {infiles} ]',
            input=f'-mergetime -apply,-sellonlatbox,{self.bbox} [ {f} ]', 
            output=self.mswep_wts
        )

    def clean(self):
        self.cdo.cleanTempDir()

if __name__ == '__main__':
    main()

