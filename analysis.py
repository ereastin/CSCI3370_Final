import xarray as xr
from dask.distributed import Client, LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/eastinev/ai')
import utils
import paths as pth
from metrics import *

# ---------------------------------------------------------------------------------
def main():
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus)
    with Client(cluster) as client:
        season = 'spr'
        model_name = 'inc3d'
        tag = 'mcs_new'
        note = f'{season}_{tag}'

        ## LSF stuff


        ## Precip stuff
        dl = xr.open_dataset(f'./DL_{season}_CESM.CTRL.nc')
        cesm = xr.open_dataset(f'./CESM_{season}_CESM.CTRL.nc')
        mswep = xr.open_dataset(f'./MSWEP_{season}_CESM.CTRL.nc')
        dl_4x = xr.open_dataset(f'./DL_{season}_CESM.2K.nc')
        cesm_4x = xr.open_dataset(f'./CESM_{season}_CESM.2K.nc')
        complete_4x = [dl_4x, cesm_4x]
        complete_ctrl = [dl, cesm, mswep]
        complete = complete_ctrl + complete_4x

        # CUS 3D Inc MERRA2 grid
        gridshape = (53, 65)  # (H, W)
        extent = (-110, -70, 25, 51)
        lons, lats = np.linspace(extent[0], extent[1], gridshape[1]), np.linspace(extent[2], extent[3], gridshape[0])

        # plotting directives
        precip_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'cmap': 'pprecip'}
        bias_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'cmap': 'bias'}
        clima_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'cmap': 'clima'}

        # steps per day e.g. (24 / spd) hourly data
        spd = 8

        #get_pdf(complete)
        #precip(complete, season, precip_plot_params, model_name, note)
        #seasonal_stats(complete, bias_plot_params, clima_plot_params, spd, model_name, note)
        #annual_stats(complete, bias_plot_params, clima_plot_params, spd, model_name, note)
        #daily_stats(complete)
        #hourly_stats(complete, bias_plot_params, clima_plot_params, spd, season, model_name, note)

# ---------------------------------------------------------------------------------
def get_pdf(complete):
    # Plot PDF of rainfall
    fig, ax = plt.subplots(1, 2)
    daily = [da.resample(time='D').sum(dim='time').dropna('time', how='all') for da in complete]
    thresh = np.arange(101)
    out = []
    # TODO: idk that this is right.. qualitatively similar to paper distr.?
    for da in daily:
        wt = surface_area(da)  # seems like the same as cos(lat) weighting?
        a = []
        for t in thresh:
            b = da.where(da > t, 0).weighted(wt).mean(dim=['lat', 'lon'], skipna=True)
            c = b.mean(dim='time').to_dataarray().to_numpy()
            a.append(c[0])
        out.append(np.asarray(a))

    colors = ['blue', 'red', 'black', 'green', 'gold']
    styles = ['solid', 'solid', 'solid', 'solid', 'solid']
    labels = ['DL', 'CESM', 'MSWEP', 'DL4x', 'CESM4x']
    for o, c, l, ls in zip(out, colors, labels, styles):
        ax[0].plot(thresh, o, color=c, label=l, linestyle=ls)
        ax[1].plot(thresh, o / o[0], color=c, label=l, linestyle=ls)

    #ax.hist(values, n_bins, histtype='step', density=True, edgecolor=colors, label=labels)
    ax[0].set(ylim=(0, 4), xlim=(0, 100), ylabel=r'$p(r > r_t)$ [mm/day]', xlabel=r'$r_t$ [mm/day]')
    ax[1].set(ylim=(0, 1), xlim=(0, 100), xlabel=r'$r_t$ [mm/day]')
    ax[1].legend()
    fig.tight_layout()
    plt.savefig('./test.png', dpi=300)

# ---------------------------------------------------------------------------------
def precip(complete, season, plot_params, model_name, note):
    # Known Sumatra Squall events (not in test apparently)
    # selected = [(2016, 7, 12), (2014, 6, 11), (2014, 6, 12)]
    match season:
        case 'both':
            selected = ['2017-08-04', '2008-05-12', '2007-04-18', '2015-07-26']
        case 'sum':
            #selected = ['2008-07-23', '2010-06-30', '2010-08-24', '2009-07-21']
            # dry days
            #selected += ['2010-06-25', '2015-07-10', '2019-07-31', '2005-07-12']
            selected = ['1983-06-25', '1980-06-19', '1983-06-19']  # highest total rain
            #selected = ['1983-06-27', '1980-07-21', '1979-06-20']
        case 'spr':
            #selected = ['2010-03-17', '2009-05-10', '2016-04-10', '2013-04-25']
            # dry days
            #selected += ['2010-03-05', '2018-04-18', '2007-04-12', '2013-03-02']
            #selected = ['1982-03-01', '1979-04-16', '1981-04-06']  # highest toal rain
            selected = ['1979-04-17', '1980-03-09', '1982-04-10']

    complete_pred, complete_obs = complete[0], complete[1]
    for dt in selected:
        # will select all present times of specified day
        pred_sel = complete_pred.sel(time=dt)['precip'].as_numpy()
        obs_sel = complete_obs.sel(time=dt)['precip'].as_numpy()
        utils.plot_precip(
            pred_sel,
            obs_sel,
            dt,
            plot_params,
            model_name,
            note=note
        )

# ---------------------------------------------------------------------------------
def annual_stats(complete, bias_plot_params, clima_plot_params, spd, model_name, note):
    # per-year daily mean
    annual = [da.groupby('time.year').mean(dim='time') * spd for da in complete]

    # total annual mean daily precip
    labels = ['DL', 'CESM']
    climatology = {}
    for da, _id in zip(annual, labels):
        climatology |= {str(yr) + _id: da.sel(year=yr)['precip'].as_numpy() for yr in da.year.values}

    utils.plot_mean(
        climatology,
        clima_plot_params,
        model_name,
        note=note + '_annual',
        bias=False
    )

    # total annual mean daily precip bias
    bias = annual[0] - annual[1]
    utils.plot_mean(
        {str(yr): bias.sel(year=yr)['precip'].as_numpy() for yr in bias.year.values},
        bias_plot_params,
        model_name,
        note=note + '_annual',
        bias=True
    )

# ---------------------------------------------------------------------------------
def daily_stats(complete):
    # resample to regional latitude-weighted daily means
    daily = [_lat_wtd(da.resample(time='D').mean(dim='time').dropna('time', how='all')) for da in complete]
    diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    print(diff.values[:8])
    print(diff.time.values[:8])
    return
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in daily]
    labels = ['DL', 'CESM']
    for item, _id in zip(min_obs, labels):
        print(_id)
        print(item.values[:8])
        print(item.time.values[:8])

# ---------------------------------------------------------------------------------
def seasonal_stats(complete, bias_plot_params, clima_plot_params, spd, model_name, note):
    seasonal = [da.groupby('time.season').mean(dim='time') * spd for da in complete]

    # total seasonal mean daily precip
    labels = ['DL', 'CESM']
    climatology = {}
    for da, _id in zip(seasonal, labels):
        climatology |= {ssn + _id: da.sel(season=ssn)['precip'].as_numpy() for ssn in da.season.values}

    utils.plot_mean(
        climatology,
        clima_plot_params,
        model_name,
        note=note + '_seasonal',
        bias=False
    )

    # total seasonal mean daily precip bias
    ctrl = xr.open_dataset('./pred_sum_ctrl.nc')
    bias = seasonal[0] - ctrl
    utils.plot_mean(
        {ssn: bias.sel(season=ssn)['precip'].as_numpy() for ssn in bias.season.values},
        bias_plot_params,
        model_name,
        note=note + '_seasonal',
        bias=True
    )

# ---------------------------------------------------------------------------------
def hourly_stats(complete, bias_plot_params, clima_plot_params, spd, season, model_name, note):
    # sub-region slices
    NGP_slat, SGP_slat, slat, slon = slice(40, 48), slice(31, 40), slice(31, 48), slice(-102, -85)

    hourly = [da.groupby('time.hour').mean(dim='time') * spd for da in complete]  # per gridcell daily mean
    NGP = [_lat_wtd(da.sel(lat=NGP_slat, lon=slon)) for da in hourly]  # order as pred, obs, (comp)
    SGP = [_lat_wtd(da.sel(lat=SGP_slat, lon=slon)) for da in hourly]

    fig, [ax0, ax1] = plt.subplots(1, 2, sharey=True)
    colors = ['blue', 'red', 'black', 'green', 'gold']
    styles = ['solid', 'solid', 'solid', 'solid', 'solid']
    labels = ['DL', 'CESM', 'MSWEP', 'DL4x', 'CESM4x']
    for da, _id, c in zip(NGP, labels, colors): da.plot.line(ax=ax0, color=c, label=_id)
    for da, _id, c in zip(SGP, labels, colors): da.plot.line(ax=ax1, color=c, label=_id)
    ax0.set(title='NGP', xlabel='UTC Hour', ylabel='mean accum. [mm/d]', xlim=(0, 21))
    ax1.set(title='SGP', xlabel='UTC Hour', ylabel='', xlim=(0, 21))
    ax0.legend(frameon=False, fancybox=False, fontsize='small')
    fig.suptitle(f'Regional Mean Diurnal Cycle for {_season(season)}')
    fig.tight_layout()
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'line_diurnal_{note}.png'), dpi=300)
    return

    # Regional plot of diurnal cycle
    utils.plot_mean(
        {'PRED ' + str((hr - 6) % 24) + 'CST': hourly_pred.sel(hour=hr)['precip'].as_numpy() for hr in hours} | {'OBS ' + str((hr - 6) % 24) + 'CST': hourly_obs.sel(hour=hr)['precip'].as_numpy() for hr in hours},
        clima_plot_params,
        model_name,
        note=note + '_diurnal',
        bias=False
    )
    utils.plot_mean(
        {'BIAS ' + str((hr - 6) % 24) + 'CST': bias.sel(hour=hr)['precip'].as_numpy() for hr in hours},
        bias_plot_params,
        model_name,
        note=note + '_diurnal_bias',
        bias=True
    )
    return

def _season(season):
    match season:
        case 'sum':
            return 'JJA'
        case 'spr':
            return 'MAM'

# ---------------------------------------------------------------------------------
def _lat_wtd(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).mean(dim=['lat', 'lon'])['precip']

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
