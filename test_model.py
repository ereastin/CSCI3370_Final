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

# custom imports
from PrecipDataset import PrecipDataset
from OTPrecipDataset import OTPrecipDataset
from Networks import *
from InceptUNet import IRNv4UNet
from InceptUNet3D import IRNv4_3DUNet
from v2 import MultiUNet

sys.path.append('/home/eastinev/AI')
import utils
import paths as pth

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
BIAS = False
CLASSIFIER = False

## ================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    parser.add_argument('-t', '--tag', type=str, default='')
    args = parser.parse_args()

    model_name = args.model_name
    exp, season = args.exp, args.season
    tag = args.tag
    model_id_tag = season + tag

    perturb_dict = {}

    note = model_id_tag + '_ctrl'  # TODO: adjust this based on perturbation experiments
    #note += 'trn'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    weekly = True
    test_loader = prep_loader(exp, season, weekly, perturb_dict)
    model = prep_model(model_name, model_id_tag, in_channels=6)
    model = load_model(model, model_name, model_id_tag, device)

    print(f'Running test experiment {note} on {model_id_tag}')
    check_accuracy(model, model_name, test_loader, device, exp, note=note)

# ---------------------------------------------------------------------------------
def prep_model(model_name, tag, in_channels=64):
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
            return IRNv4UNet(in_channels)
        case 'inc3d':
            return IRNv4_3DUNet(in_channels, depth=16, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act)
        case 'unet':
            return MultiUNet(n_vars=5, depth=16, init_c=64, embedding_dim=32, bias=True)
        case _:
            print(f'Model "{model_name}" not valid')
            sys.exit(-21)

# ---------------------------------------------------------------------------------
def prep_loader(exp, season, weekly, perturb_dict):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    test_ds = OTPrecipDataset('test', exp, season, weekly, False, perturb_dict)
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=n_workers, pin_memory=False)

    return test_loader

# ---------------------------------------------------------------------------------
def load_model(model, model_name, tag, device):
    path = os.path.join(f'./models/{model_name}/params_{tag}.pth')
    model.load_state_dict(torch.load(path, map_location=device))
 
    return model

# ---------------------------------------------------------------------------------
def check_accuracy(model, model_name, loader, device, exp, note=''):
    gridshape = (54, 42)  # (H, W)
    accum_dict = {
        'DJF': [torch.zeros(gridshape)] * 3 + [0],
        'MAM': [torch.zeros(gridshape)] * 3 + [0],
        'JJA': [torch.zeros(gridshape)] * 3 + [0],
        'SON': [torch.zeros(gridshape)] * 3 + [0],
    }

    # Sumatra Squall MERRA2
    # extent = (92.8, 107.2, 0.0, 8.0)

    # CUS 3D Inc MERRA2
    # extent = (-140, -50.625, 14, 53.5)  # full repr
    extent = (-108.75, -83.125, 24, 50.5)  # CUS ROI

    # for zhang_repr CUS 2D Inc ERA5
    # extent = (-108, -84.25, 24, 47.75)

    lons, lats = np.linspace(extent[0], extent[1], gridshape[1]), np.linspace(extent[2], extent[3], gridshape[0])
    plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'cmap': 'vis_precip'}

    # select specific (or random) days
    selected = [(1984, 6, 13), (1983, 6, 15), (1982, 4, 20), (2007, 3, 11), (1989, 7, 11)]  #, (1987, 6, 13)] overfit test
    # selected = [(np.random.choice(np.arange(1980, 2021)), np.random.choice(np.arange(4, 12)), np.random.choice(np.arange(1, 31))) for _ in range(4)]  # select 4 random days
    # Known Sumatra Squall events (not in test apparently)
    # selected = [(2016, 7, 12), (2014, 6, 11), (2014, 6, 12)]
    spd = 8

    # For un-standardizing predicted/target precip data
    with open(f'./{exp}_norm_vars.json', 'r') as f:
        stats = json.load(f)
    mn_p = stats['precipitationmn']
    std_p = stats['precipitationstd']

    test_loss = 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (source, target, time_id) in enumerate(loader):
            print(f'Testing model on {utils.get_time_str(time_id)}...')
            source, target = source.to(device=device), target.to(device=device)
            pred = model(source)
            pred, target = pred * std_p + mn_p, target * std_p + mn_p
            loss = F.mse_loss(pred, target)
            test_loss += loss.item()

            if BIAS:
                season = utils.get_season_str(time_id)
                pred_batch_accum, n_days = utils.calc_batch_accum(pred, steps_per_day=spd)
                target_batch_accum, _ = utils.calc_batch_accum(target, steps_per_day=spd)
                bias_batch_accum, _ = utils.calc_batch_accum(pred - target, steps_per_day=spd)
                accum_dict[season][0] += pred_batch_accum.numpy(force=True)
                accum_dict[season][1] += target_batch_accum.numpy(force=True)
                accum_dict[season][2] += bias_batch_accum.numpy(force=True)
                accum_dict[season][3] += n_days

            # do precip prediction plotting
            day_sel = utils.match_time_str(time_id, selected)
            for d in day_sel:
                time_sel = utils.get_time_str(time_id, day_sel)
                # fetch specific day from batch
                pred_sel = pred[d * spd:(d + 1) * spd]
                target_sel = target[d * spd:(d + 1) * spd]
                utils.plot_precip(pred_sel, target_sel, time_sel, plot_params, model_name, note=note)

    test_loss /= len(loader)
    print(f'Mean MSE loss on test set: {test_loss}')
    if BIAS:
        # per-cell average daily values
        mean_daily_bias = {k: v[2] / v[3] for k, v in accum_dict.items() if v[3] != 0}
        utils.plot_bias(mean_daily_bias, plot_params, model_name, note=note)

# ---------------------------------------------------------------------------------
if __name__ == '__main__': 
    main()

