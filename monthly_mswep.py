import os
import sys
from cdo import *
#import time

def main():
    #t1 = time.time()
    year, month = int(sys.argv[1]), int(sys.argv[2])
    mm = MakeMonthly()
    #years = [2001]  # list(range(2001, 2021))
    #months = [1]  # list(range(12)):
    try:
        #for year in years:
        #    for month in months:
        day_rng = mm.get_mswep_t(year, month)
        mm.get_target(year, month, day_rng)
    except Exception as e:
        print(f'{year}-{month} Failed')
        mm.clean()

    #t2 = time.time()
    print(f'{year}-{month} Completed')
    mm.clean()

class MakeMonthly:
    def __init__(self):
        self.wts_gen = False 
        self.target_dir = '/projects/bccg/andromeda1/bccg/Shared_Data/OBS/MSWEP/Past/3hourly/'
        self.scratch_dir = '/scratch/eastinev/'
        
        self.cdo = Cdo(tempdir=self.scratch_dir)
        self.cdo.debug = True
        #self.cdo.cleanTempDir()  # make sure to do this after completing, leave here for debugging now

        self.wts_file = os.path.join(self.scratch_dir, 'mswep_weights.nc') 
        self.dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.gfile = os.path.join(self.scratch_dir, 'mswep_gridfile.txt')
        self.gfile_txt = [
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
        self.write_gfile()

    def write_gfile(self):
        if not os.path.exists(self.gfile):
            with open(self.gfile, 'w+') as f:
                f.write('\n'.join(self.gfile_txt))

    def get_mswep_t(self, year, month):
        leap_year = (year % 4 == 0)
        days = self.dpm[month]
        days += 1 if leap_year and month == 1 else 0
        return list(range(1, days + 1))

    def get_target(self, year, month, day_rng):
        # do merge and remap
        t_strt = '00:00:00'  # combining month will always start at 00:00:00
        d_strt = f'{str(year)}-{str(month + 1).zfill(2)}-01'  # this should always start as YYYY-MM-01
        outfile = os.path.join(self.scratch_dir, f'P.{str(year)}{str(month + 1).zfill(2)}.nc')
        #if os.path.exists(outfile):
        #    return
        fnames = [f'{str(year)}{str(sum(self.dpm[:month]) + day).zfill(3)}*' for day in day_rng]
        files = [os.path.join(self.target_dir, f) for f in fnames] 
        #if not self.wts_gen:
        #    self.cdo.gencon(self.gfile, input=f'-sellonlatbox,-110,-80,20,55 {files[0][:-1] + ".00.nc"}', output=self.wts_file)
        #    self.wts_gen = True
        
        files = ' '.join(files)
        # this remapping is wayy slower like 5 mins vs 2 secs for one month... not sure there's anything to be done about it think its just slow going this way..?
        self.cdo.remap(self.gfile, self.wts_file, input=f'-inttime,{d_strt},{t_strt},6hour -mergetime -apply,-sellonlatbox,-110,-80,20,55 [ {files} ]', output=outfile)
        return

    def clean(self):
        self.cdo.cleanTempDir()

if __name__ == '__main__':
    main()

