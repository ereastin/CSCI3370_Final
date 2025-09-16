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
import seaborn as sns
import time
import json
import dask
from dask import array
from dask.distributed import Client, LocalCluster
import pandas as pd

# custom imports
from PrecipDataset import PrecipDataset
from OTPrecipDataset import OTPrecipDataset
from Networks import *
from InceptUNet import IRNv4UNet
from InceptUNet3D import IRNv4_3DUNet
from v2 import MultiUNet
from simple import Simple
import mcs_data as mcs
import file_utils as futils
from analysis import *

sys.path.append('/home/eastinev/ai')
import utils
import paths as pth

CUS_LON = slice(-110 - 1, -70 + 1)
CUS_LAT = slice(51 + 1, 25 - 1)  # mswep has these backwards, need buffer for remap
MERRA_VARS = ['QV', 'U', 'V', 'OMEGA', 'H', 'T']
LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])

COMP = False
HOURLY = False
DAILY = False
SEASONAL = False
ANNUAL = False
PRECIP = False
PDF = False
MCS = False
RET_AS_TNSR = False
CESM = True
CESM_EXP = '.2K'
# these dont seem to matter for testing..?
C, D, H, W = 6, 28, 81, 145
Q_SCALE_SUM = np.array([0.73947986, 0.79641958, 0.82545568, 0.85676614, 0.8690772,  0.85588648,
 0.85501163, 0.83975696, 0.80672031, 0.77467831, 0.7172267,  0.67628894,
 0.5605,     0.53792147, 0.51062403, 0.47532429, 0.43959246, 0.37773203,
 0.36366254, 0.61924349, 0.72442247, 0.73853947, 0.7590272,  0.76995612,
 0.77492441, 0.78310098, 0.7887769,  0.80430977]).reshape(1, 28, 1, 1)
Q_SCALE_SPR = np.array([0.73484835, 0.7741839,  0.79289228, 0.78702883, 0.77525633, 0.75809729,
 0.767235,   0.75190734, 0.73003483, 0.68878793, 0.64439483, 0.61227615,
 0.54781833, 0.52006356, 0.4883695,  0.44912579, 0.40702892, 0.35720196,
 0.36740634, 0.63666115, 0.76133438, 0.78194161, 0.77004798, 0.76141909,
 0.76900295, 0.78955253, 0.79839702, 0.81697179])
'''
## 4x - ctrl stuff
pred = xr.open_dataset('./pred_sum_ctrl.nc')
pred4x = xr.open_dataset('./pred_sum_4k.nc')
bias = pred4x - pred

# total seasonal mean daily precip bias
utils.plot_mean(
    {ssn: bias.sel(season=ssn)['precip'].as_numpy() for ssn in bias.season.values},
    bias_plot_params,
    model_name,
    note='pred_sum_4x-ctrl_seasonal',
    bias=True
)
return

# 2012 - climatology stuff
p2012 = xr.open_dataset('./pred2012.nc')
pssn = xr.open_dataset('./predSSN.nc').sel(season='JJA')
o2012 = xr.open_dataset('./obs2012.nc')
ossn = xr.open_dataset('./obsSSN.nc').sel(season='JJA')
pdiff = p2012 - pssn
odiff = o2012 - ossn
utils.plot_mean(
    {'2012 - JJA Clima, PRED': pdiff['precip'].as_numpy()} | {'2012 - JJA Clima, OBS': odiff['precip'].as_numpy()},
    bias_plot_params,
    model_name,
    note='2012_climadiff',
    bias=True
)
return
'''

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

    test_loader = prep_loader(exp, season)
    model = prep_model(model_name, model_id_tag)
    model = load_model(model, model_name, model_id_tag, device)

    if CESM: model_id_tag += f'_CESM{CESM_EXP}'
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

        low-level: 1000-925 hPa
        mid-level: 925-500 hPa
        LLJ: approx as 850 hPa winds (?)
        '''
        perturb_dict = {
            0: {
                'var': 'Q',
                'type': 'scale',
                'scale': Q_SCALE_SUM,
                'invert': False,
                'levels': None,
            },
        }
        perturb_dict = {}
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

# ---------------------------------------------------------------------------------
def check_accuracy(model, model_name, loader, device, exp, season, note='', perturb_dict={}, df_list=None):
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

    # For un-standardizing predicted/target precip data
    with open(f'./cus_norm_vars_{season}_mcs.json', 'r') as f:
        stats = json.load(f)
        # for using CESM data
        stats['Z3mn'] = stats['Hmn']
        stats['Z3std'] = stats['Hstd']
        stats['Qmn'] = stats['QVmn']
        stats['Qstd'] = stats['QVstd']
        stats['PRECTmn'] = stats['precipitationmn']
        stats['PRECTstd'] = stats['precipitationstd']
    mn_p = stats['precipitationmn']
    std_p = stats['precipitationstd']

    # output storage
    all_preds, all_obs = [], []
    all_comp = []
    losses, pccs, ets = [], [], []

    model = model.to(device)
    model.eval()
    print('*' * 60, flush=True)
    print(f'Perturbation Experiment: {perturb_dict}', flush=True)
    with torch.no_grad():
        for i, (source_ds, target_ds, time_id) in enumerate(loader):
            if source_ds is None or target_ds is None: continue

            print(f'Testing model on {time_id[0].astype(str)}-{time_id[-1].astype(str)}...')
            # save coordinates of target for reuse
            out_coords = target_ds.coords

            if MCS:
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

            # standardize after perturbation
            for v in source_ds.variables:
                if v in ['time', 'lat', 'lon', 'plev', 'lev']:
                    continue
                source_ds[v] = (source_ds[v] - stats[v + 'mn']) / stats[v + 'std']
            for v in target_ds.variables:
                if v in ['time', 'lat', 'lon', 'plev', 'lev']:
                    continue
                target_ds[v] = (target_ds[v] - mn_p) / std_p

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

            # store in DataArray
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

            if COMP:
                # TODO: how does this work with no leap year..?
                # just skips feb 29.? other days the 'same' tho? accessing by day of year seems wrong then?
                # load mswep and regrid:
                mswep_fs = futils._get_mswep_by_time(time_id, forecast_step=1)
                mswep_ds = futils.read_target(
                    mswep_fs,
                    [],
                    {
                        'do': True,
                        'target_grid': '~/AI/incept/pgrid.nc',
                        'regrid_type': 'conservative'
                    },
                    lat=CUS_LAT,
                    lon=CUS_LON
                )
                mswep_da = mswep_ds['precipitation'].rename('precip').compute()
                all_comp.append(mswep_da)

    # Compile test performance metrics
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

    ## Output testing
    # Compile all predicted and observed precip data
    complete_pred = xr.merge(all_preds)
    complete_obs = xr.merge(all_obs)

    # this handles the forecasting step leakage into other seasons by dropping..
    # do custom seasons instead? or set back input fields instead of set forward precip field?
    if season == 'sum':
        tmask = ~((complete_pred.time.dt.month == 9) & (complete_pred.time.dt.day == 1))
    elif season == 'spr':
        tmask = ~((complete_pred.time.dt.month == 6) & (complete_pred.time.dt.day == 1))

    complete_pred = complete_pred.sel(time=tmask)
    complete_obs = complete_obs.sel(time=tmask)
    complete = [complete_pred, complete_obs]

    if COMP:
        complete_comp = xr.merge(all_comp)
        complete_comp = complete_comp.sel(time=tmask)
        complete.append(complete_comp)

    # TODO: do this for ~relevant~ perturbations as well then can easily reproduce
    #complete_pred.to_netcdf(f'./DL_{season}_CESM{CESM_EXP}.nc', engine='netcdf4')
    #complete_obs.to_netcdf(f'./CESM_{season}_CESM{CESM_EXP}.nc', engine='netcdf4')
    #complete_comp.to_netcdf(f'./MSWEP_{season}_CESM{CESM_EXP}.nc', engine='netcdf4')

    # Plot precip from selected days
    if PRECIP:
        precip(complete, season, precip_plot_params, model_name, note)

    # Hourly statistics
    if HOURLY:
        hourly_stats(complete, bias_plot_params, clima_plot_params, spd, season, model_name, note)

    # Daily statistics
    if DAILY:
        daily_stats(complete)

    # Seasonal statistics
    if SEASONAL:
        seasonal_stats(complete, bias_plot_params, clima_plot_params, spd, model_name, note)

    # Annual statistics
    if ANNUAL:
        annual_stats(complete_pred, complete_obs, bias_plot_params, clima_plot_params, spd, model_name, note)

# ---------------------------------------------------------------------------------
def feature_importance_stats(test_loss, all_ets, exp, season, perturb_dict, df_list):
    # RMSE/ETS importance metrics:
    # to save rmse and ets for each test item
    #df = pd.DataFrame({'rmse': np.sqrt(test_loss), 'ets': all_ets})
    #df.to_csv(f'./{exp}_{season}_test_stats.csv')
    #return
    ctrl_df = pd.read_csv(f'./{exp}_{season}_test_stats.csv')
    ctrl_loss = ctrl_df['rmse'].to_numpy()
    ctrl_ets = ctrl_df['ets'].to_numpy()
    Irmse = (np.sqrt(test_loss) - ctrl_loss) / ctrl_loss
    a1, b1, c1, d1 = calc_importance_stats(Irmse)
    Iets = (ctrl_ets - all_ets) / (ctrl_ets + 1e-5)  # in case there are legit 0s.. why.? will these screw up ets metrics?
    a2, b2, c2, d2 = calc_importance_stats(Iets)
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
    },
    index=[df_index])
    df_list.append(df)
    return df_list

# ---------------------------------------------------------------------------------
def run_feature_importance(test_var):
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
def calc_importance_stats(arr):
    mn, md = np.mean(arr), np.median(arr)
    iqr25, iqr75 = np.percentile(arr, 25), np.percentile(arr, 75)
    print(mn, md, iqr25, iqr75, flush=True)
    return mn, md, iqr25, iqr75

# ---------------------------------------------------------------------------------
def prep_model(model_name, tag):
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
        case 'unet':
            return MultiUNet(n_vars=C, depth=D, init_c=64, embedding_dim=32, bias=True)
        case _:
            print(f'Model "{model_name}" not valid')
            sys.exit(-21)

# ---------------------------------------------------------------------------------
def prep_loader(exp, season):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK']) if RET_AS_TNSR else 0
    test_ds = OTPrecipDataset('test', exp, season, standardize=False, shuffle=False, ret_as_tnsr=RET_AS_TNSR, cesm=CESM, cesm_exp=CESM_EXP)
    # probs need to add this in training too
    def collate(batch):
        return batch
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=n_workers, collate_fn=collate, pin_memory=False)

    return test_loader

# ---------------------------------------------------------------------------------
def load_model(model, model_name, tag, device):
    path = os.path.join(f'./models/{model_name}/params_{tag}.pth')
    model.load_state_dict(torch.load(path, map_location=device))
 
    return model

# ---------------------------------------------------------------------------------
if __name__ == '__main__': 
    main()

