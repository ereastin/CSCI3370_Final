## test_model.py: testing framework for proposed models
# ---------------------------------------------------------------------------------
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# std lib imports
import os
import sys
from argparse import ArgumentParser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json
from scipy.special import inv_boxcox
import dask
from dask import array
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import pandas as pd
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom imports
from PrecipDataset import PrecipDataset
from OTPrecipDataset import OTPrecipDataset
from Networks import *
from InceptUNet import IRNv4UNet
from InceptUNet3D import IRNv4_3DUNet
from v2 import MultiUNet
from simple import Simple
import mcs_data as mcs

sys.path.append('/home/eastinev/AI')
import utils
import paths as pth

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MERRA_VARS = ['QV', 'U', 'V', 'OMEGA', 'H', 'T']
LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])
BIAS = True
PRECIP = False
MCS = True
RET_AS_TNSR = False
C, D, H, W = 6, 8, 80, 144

## ================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-v', '--var', type=str, default='')
    args = parser.parse_args()

    model_name = args.model_name
    exp, season = args.exp, args.season
    tag = args.tag
    test_var = args.var
    model_id_tag = season + tag

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    weekly = True
    test_loader = prep_loader(exp, season, weekly)
    model = prep_model(model_name, model_id_tag)
    model = load_model(model, model_name, model_id_tag, device)

    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus)
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        '''
        Song 2019:
        Here, we use the zonal and meridional winds at three levels (925, 500, and 200 hPa)
        and the specific humidity at two levels (925 and 500 hPa)
        (and geopotential at these levels)

        Feng 2019:
        large-scale 925-hPa wind and specific humidity, and 500-hPa geopotential height
        '''
        perturb_dict = {
            0: {
                'var': 'QV',
                'type': 'scale',
                'scale': 0,
                'invert': False,
                'levels': None,
            },
        }
        check_accuracy(
            model,
            model_name,
            test_loader,
            device,
            exp,
            season,
            note=model_id_tag,
            perturb_dict=perturb_dict,
        ) 

def shuffle(test_var):
    df_list = []
    try:
        for v in [test_var]:
            for l in LEV:
                perturb_dict = {
                    0: {
                        'var': v,
                        'type': 'shuffle',
                        'scale': 1,
                        'invert': False,
                        'levels': l,
                    }
                }
                df_list = check_accuracy(
                    model,
                    model_name,
                    test_loader,
                    device,
                    exp,
                    season,
                    note=model_id_tag,
                    perturb_dict=perturb_dict,
                    df_list=df_list
                )

            out_df = pd.concat(df_list)
            print(out_df)
            out_df.to_csv(f'./{exp}_{season}_{v}_shuffle_add.csv')
    except KeyboardInterrupt:
        out_df = pd.concat(df_list)
        print(out_df)
        out_df.to_csv(f'./{exp}_{season}_{v}_shuffle_add.csv')

# ---------------------------------------------------------------------------------
def prep_model(model_name, tag):
    # TODO: just pass this to the model and have it load from there
    path = os.path.join(f'./models/{model_name}/hyperparams_{tag}.json')
    with open(path, 'r') as f:
        hps = json.load(f)
    print(f'Loading hyperparams for model {model_name}:\n {hps}')
    Na, Nb, Nc = hps['Na'], hps['Nb'], hps['Nc']
    base = hps['base']
    lin_act = hps['lin_act']
    bias = hps['bias']
    drop_p = hps['drop_p']

    # create specified model
    match model_name:
        case 'inc':
            return IRNv4UNet(C)
        case 'inc3d':
            return Simple(C, depth=D, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act)
            #return IRNv4_3DUNet(C, depth=D, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act)
        case 'unet':
            return MultiUNet(n_vars=C, depth=D, init_c=64, embedding_dim=32, bias=True)
        case _:
            print(f'Model "{model_name}" not valid')
            sys.exit(-21)

# ---------------------------------------------------------------------------------
def prep_loader(exp, season, weekly):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK']) if RET_AS_TNSR else 0
    test_ds = OTPrecipDataset('test', exp, season, weekly, shuffle=False, ret_as_tnsr=RET_AS_TNSR)
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=n_workers, pin_memory=False)

    return test_loader

# ---------------------------------------------------------------------------------
def load_model(model, model_name, tag, device):
    path = os.path.join(f'./models/{model_name}/params_{tag}.pth')
    model.load_state_dict(torch.load(path, map_location=device))
 
    return model

# ---------------------------------------------------------------------------------
def check_accuracy(model, model_name, loader, device, exp, season, note='', perturb_dict={}, df_list=None):
    gridshape = (53, 65) #(54, 42)  # (H, W)

    # Sumatra Squall MERRA2
    # extent = (92.8, 107.2, 0.0, 8.0)

    # CUS 3D Inc MERRA2
    # extent = (-140, -50.625, 14, 53.5)  # full repr
    # extent = (-108.75, -83.125, 24, 50.5)  # CUS ROI
    extent = (-110, -70, 25, 51)

    # for zhang_repr CUS 2D Inc ERA5
    # extent = (-108, -84.25, 24, 47.75)

    lons, lats = np.linspace(extent[0], extent[1], gridshape[1]), np.linspace(extent[2], extent[3], gridshape[0])
    precip_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'figsize': (16, 4), 'cmap': 'pprecip'}
    bias_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'figsize': (8, 8), 'cmap': 'bias'}
    clima_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'figsize': (16, 4), 'cmap': 'clima'}

    # select specific (or random) days
    match season:
        case 'both':
            selected = [(2017, 8, 4), (2008, 5, 12), (2007, 4, 18), (2015, 7, 26)]
        case 'sum':
            selected = [(2004, 6, 9), (2010, 7, 11), (2015, 7, 27), (2015, 7, 9)]
        case 'spr':
            selected = [(2010, 3, 17), (2009, 5, 10), (2016, 4, 10), (2013, 4, 25)]

    # Known Sumatra Squall events (not in test apparently)
    # selected = [(2016, 7, 12), (2014, 6, 11), (2014, 6, 12)]
    spd = 8

    # For un-standardizing predicted/target precip data
    with open(f'./{exp}_norm_vars_{season}_mcs.json', 'r') as f:
        stats = json.load(f)
    mn_p = stats['precipitationmn']
    std_p = stats['precipitationstd']

    all_preds, all_obs = [], []
    losses, pccs, ets = [], [], []
    model = model.to(device)
    model.eval()
    print('*' * 60, flush=True)
    print(f'Perturbation Experiment: {perturb_dict}', flush=True)
    with torch.no_grad():
        for i, (source_ds, target_ds, time_id) in enumerate(loader):
            #print(f'Testing model on {utils.get_time_str(time_id)}...')
            out_coords = target_ds.coords

            if MCS:
                # TODO: was this ever adjusted to match correct times of MERRA data..?
                # get mask and valid times
                mask, times = mcs.run(time_id)
                # add MCS CCS mask to perturb_dict
                if perturb_dict != {}:
                    for k in perturb_dict.keys():
                        perturb_dict[k]['region'] = mask
            else:
                if perturb_dict != {}:
                    for k in perturb_dict.keys():
                        perturb_dict[k]['region'] = None

            # identify w experiment type
            if i == 0:
                note += utils.build_exp_note(perturb_dict)

            # do input perturbation
            source_ds = utils.perturb(source_ds, perturb_dict)

            # normalize after perturbation
            for v in MERRA_VARS:
                source_ds[v] = (source_ds[v] - stats[v + 'mn']) / stats[v + 'std']
            pv = 'precipitation'
            target_ds[pv] = (target_ds[pv] - mn_p) / std_p

            # compute and close open Datasets
            source = source_ds.to_dataarray().to_numpy()
            source_ds.close()
            source = torch.tensor(source).permute(1, 0, 2, 3, 4) # return as (time, var, (lev), lat, lon) 
            target = target_ds.to_dataarray().to_numpy()
            target_ds.close()
            target = torch.tensor(target).permute(1, 0, 2, 3) # return as (time, var, lat, lon) 

            # send to device and predict
            source, target = source.to(device=device), target.to(device=device)
            pred = model(source)
            # un-standardize
            pred, target = pred * std_p + mn_p, target * std_p + mn_p

            # accumulate loss
            loss = torch.mean((pred - target) ** 2, dim=(2, 3))
            losses.append(loss)

            # grab test metrics
            pccs.append(utils.cum_pcc(pred, target))
            ets.append(utils.cum_ets(pred, target))

            pred_da = xr.DataArray(
                data=pred.squeeze(1).numpy(force=True),
                dims=('time', 'lat', 'lon'),
                coords=out_coords,
                name='precip'
            )
            obs_da = xr.DataArray(
                data=target.squeeze(1).numpy(force=True),
                dims=('time', 'lat', 'lon'),
                coords=out_coords,
                name='precip'
            )
            all_preds.append(pred_da)
            all_obs.append(obs_da)

            # do precip prediction plotting
            # TODO: can for sure fix this with xarray stuff
            if PRECIP:
                # TODO: fix this to grab correct days when selecting just for MCS days
                day_sel = utils.match_time_str(time_id, selected)
                for d in day_sel:
                    time_sel = utils.get_time_str(time_id, day_sel)
                    # fetch specific day from batch
                    pred_sel = pred[d * spd:(d + 1) * spd]
                    target_sel = target[d * spd:(d + 1) * spd]
                    utils.plot_precip(
                        pred_sel,
                        target_sel,
                        time_sel,
                        precip_plot_params,
                        model_name,
                        note=note
                    )

    # Compile all predicted and observed precip data
    complete_pred = xr.merge(all_preds)
    complete_obs = xr.merge(all_obs)

    # TODO: possibly easier way to manage this re: bias vs. just pred/obs accum, selecting between them on CLI
    # Seasonal statistics -- mean daily (bias)
    seasonal_pred = complete_pred.groupby('time.season').mean(dim='time') * spd
    seasonal_obs = complete_obs.groupby('time.season').mean(dim='time') * spd
    bb = (seasonal_pred - seasonal_obs)
    # total seasonal mean daily precip bias
    utils.plot_mean(
        {ssn: bb.sel(season=ssn)['precip'].as_numpy() for ssn in bb.season.values},
        bias_plot_params,
        model_name,
        note=note,
        bias=True
    )
    # total seasonal mean daily precip
    utils.plot_mean(
        {ssn + ' PRED': seasonal_pred.sel(season=ssn)['precip'].as_numpy() for ssn in seasonal_pred.season.values} | {ssn + ' OBS': seasonal_obs.sel(season=ssn)['precip'].as_numpy() for ssn in seasonal_obs.season.values},
        clima_plot_params,
        model_name,
        note=note,
        bias=False
    )


    # Hourly statistics
    hourly_pred = complete_pred.groupby('time.hour').mean(dim='time') * spd
    hourly_obs = complete_obs.groupby('time.hour').mean(dim='time') * spd

    # regional mean daily mean.
    weight = np.cos(np.radians(hourly_pred.lat))
    hov_pred = hourly_pred.weighted(weight).mean(dim='lat')
    # TODO: handle cyclic coord and plotting with midnight in the middle
    #new_hour = (hov_pred.hour - 6) % 24
    #hov_pred['hour'] = new_hour
    #hov_pred = hov_pred.roll(hour=2, roll_coords=True)
    #print(hov_pred)
    hov_pred['precip'].plot.pcolormesh()
    plt.savefig(f'./{season}_hovpred_{note}.png')
    plt.clf()
    plt.close()
    hov_obs = hourly_obs.weighted(weight).mean(dim='lat')
    #hov_obs['hour'] = new_hour
    #hov_obs = hov_obs.roll(hour=2, roll_coords=True)
    #print(hov_obs)
    hov_obs['precip'].plot.pcolormesh()
    plt.savefig(f'./{season}_hovobs_{note}.png')
    plt.clf()
    plt.close()

    # TODO: cleaner way to do this?
    utils.plot_mean(
        {'PRED ' + str((t - 6) % 24) + 'CST': hourly_pred.sel(hour=t)['precip'].as_numpy() for t in hourly_pred.hour.values} | {'OBS ' + str((t - 6) % 24) + 'CST': hourly_obs.sel(hour=t)['precip'].as_numpy() for t in hourly_obs.hour.values},
        clima_plot_params,
        model_name,
        note=note + '_diurnal',
        bias=False
    )

    test_loss = torch.cat(losses, dim=0).numpy(force=True).flatten()
    all_pccs = torch.cat(pccs, dim=0).numpy(force=True).flatten()
    good_pccs = np.sum(np.where(all_pccs >= 0.7, 1, 0))
    test_pcc = np.mean(all_pccs)
    all_ets = torch.cat(ets, dim=0).numpy(force=True).flatten()
    test_ets = np.mean(all_ets)
    print(f'Per-item mean MSE: {np.mean(test_loss)}', flush=True)
    print(f'Per-item mean centered PCC: {test_pcc}', flush=True)
    print(f'Items w/PCC > 0.7: {good_pccs} of {len(all_pccs)}, {good_pccs / len(all_pccs) * 100:.2f}%', flush=True)
    print(f'Per-item mean ETS: {test_ets}', flush=True)

    return

    # NOTE: used for shuffled feature importance
    # RMSE/ETS importance metrics:
    # to save rmse and ets for each test item
    #df = pd.DataFrame({'rmse': np.sqrt(test_loss), 'ets': all_ets})
    #df.to_csv(f'./{exp}_{season}_test_stats.csv')
    #return
    ctrl_df = pd.read_csv(f'./{exp}_{season}_test_stats.csv')
    ctrl_loss = ctrl_df['rmse'].to_numpy()
    ctrl_ets = ctrl_df['ets'].to_numpy()
    Irmse = (np.sqrt(test_loss) - ctrl_loss) / ctrl_loss
    a1, b1, c1, d1 = calc_stats(Irmse)
    Iets = (ctrl_ets - all_ets) / (ctrl_ets + 1e-5)  # in case there are legit 0s.. why.? will these screw up ets metrics?
    a2, b2, c2, d2 = calc_stats(Iets)
    df_index = perturb_dict[0]['var'] + str(perturb_dict[0]['levels'])
    df = pd.DataFrame({
        'mean_mse': np.mean(test_loss),
        'mean_pcc': test_pcc,
        'n_good_pcc': good_pccs,
        'mean_ets': test_ets,
        'rmse_imp_mean': a1,
        'rmse_imp_med': b1,
        'rmse_imp_25p': c1,
        'rmse_imp_75p': d1,
        'ets_imp_mean': a2,
        'ets_imp_med': b2,
        'ets_imp_25p': c2,
        'ets_imp_75p': d2,
        'mean_reg_bias': mb
    },
    index=[df_index])
    df_list.append(df)
    return df_list

def calc_stats(arr):
    mn, md = np.mean(arr), np.median(arr)
    iqr25, iqr75 = np.percentile(arr, 25), np.percentile(arr, 75)
    print(mn, md, iqr25, iqr75, flush=True)
    return mn, md, iqr25, iqr75

# ---------------------------------------------------------------------------------
if __name__ == '__main__': 
    main()

