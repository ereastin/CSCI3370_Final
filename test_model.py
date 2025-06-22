## train.py: training framework for proposed models
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

# custom imports
from PrecipDataset import PrecipDataset
from OTPrecipDataset import OTPrecipDataset
from Networks import *
from InceptUNet import IRNv4UNet
from InceptUNet3D import IRNv4_3DUNet

sys.path.append('/home/eastinev/AI')
import utils
import paths as pth

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
## ================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('-r', '--rmvar', type=str)
    parser.add_argument('-z', '--zero', action='store_true')
    parser.add_argument('-p', '--precip', action='store_true')
    parser.add_argument('-b', '--bias', action='store_true')
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    parser.add_argument('-t', '--tag', type=str, default='')
    args = parser.parse_args()

    model_name = args.model_name
    rm_var, zero = args.rmvar, args.zero
    exp, season = args.exp, args.season
    tag = args.tag
    # TODO: what about removing vertical shear? set all to..? average of midlevel?
    model_id_tag = season + tag
    note = model_id_tag + '_ctrl' if rm_var is None else f'{"no" if zero else "mn"}{rm_var}'
    #note += 'trn'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    test_loader = prep_loader(exp, season, rm_var, zero, weekly=True)
    model = prep_model(model_name, model_id_tag, in_channels=6)
    model = load_model(model, model_name, model_id_tag, device)

    print(f'Running test experiment {note} on {model_id_tag}')
    check_accuracy(model, model_name, test_loader, device, note=note, precip=args.precip, bias=args.bias)

# ---------------------------------------------------------------------------------
def prep_model(model_name, tag, in_channels=64):
    # TODO: just pass this to the model and have it load from there
    path = os.path.join(f'./models/{model_name}/hyperparams_{tag}.json')
    with open(path, 'r') as f:
        hps = json.load(f)
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
            return IRNv4_3DUNet(in_channels, depth=35, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act)
        case _:
            print(f'Model "{model_name}" not valid')
            sys.exit(-21)

# ---------------------------------------------------------------------------------
def prep_loader(exp, season, rm_var, zero, weekly=False):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    test_ds = OTPrecipDataset('test', exp, season, weekly, rm_var, zero)
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=n_workers, pin_memory=False)

    return test_loader

# ---------------------------------------------------------------------------------
def load_model(model, model_name, tag, device):
    path = os.path.join(f'./models/{model_name}/params_{tag}.pth')
    model.load_state_dict(torch.load(path, map_location=device))
 
    return model

# ---------------------------------------------------------------------------------
def check_accuracy(model, model_name, loader, device, note='', precip=False, bias=False):
    loss_list, monthly_bias = [], []
    selected = []
    d_pred, d_obs, d_bias = [], [], []
    rand_weeks = torch.randint(len(loader) - 1, (4,)).tolist()  # select 6 random weeks

    # sumatra squall stuff
    extent = (92.8, 107.2, 0.0, 8.0)
    gridshape = (80, 144)  # (H, W)
    # dlon, dlat = 0.1, 0.1  # np.arange() isnt stable when start >> step for some reason..
    
    # CUS 3D Inc MERRA2
    # extent = (-140, -50.625, 14, 53.5)  # full repr
    # extent = (-108.75, -83.125, 24, 50.5)  # CUS ROI
    # dlon, dlat = 0.625, 0.5

    # for zhang_repr CUS 2D Inc
    # extent = (-108, -84.25, 24, 47.75)
    # dlon, dlat = 0.25, 0.25

    lons, lats = np.linspace(extent[0], extent[1], gridshape[1]), np.linspace(extent[2], extent[3], gridshape[0])
    plot_params = {'extent': extent, 'lons': lons, 'lats': lats}

    # select specific days
    selected = [(1989, 4, 12), (1980, 11, 22), (1988, 4, 24), (2006, 5, 18)]
    # selected = [(1996, 6, 26), (2003, 5, 20), (1993, 5, 10), (2009, 4, 29)]
    # Sumatra Squall events (not in test apparently)
    # selected = [(2016, 7, 12), (2014, 6, 11), (2014, 6, 12)]

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (source, target, time_id) in enumerate(loader):
            print(f'Testing model on {utils.get_time_str(time_id)}...')
            if source is None or target is None:
                continue
            source, target = source.to(device=device), target.to(device=device)
            pred = model(source)
            loss = F.mse_loss(pred, target) # + torch.mean(F.relu(-1 * output))  # include relu in MSE calcs..?
            loss_list.append(loss.item())

            # get pdf stuff
            # pred_accum, obs_accum, bb = utils.calc_daily_accum(pred, target)
            # d_pred.append(pred_accum)
            # d_obs.append(obs_accum)
            # d_bias.append(bb)

            if bias:
                b = utils.calc_daily_bias(pred, target, steps_per_day=8)
                monthly_bias.append(b)

            day_sel = utils.match_time_str(time_id, selected)
            for d in day_sel:
                time_sel = utils.get_time_str(time_id, day_sel)
                # fetch specific day from batch
                pred_sel = pred[d * 8:(d + 1) * 8]
                target_sel = target[d * 8:(d + 1) * 8]
                utils.plot_precip(pred_sel, target_sel, time_sel, plot_params, model_name, note=note)

    mean_loss = np.mean(loss_list)
    print(f'Mean MSE loss on test set: {mean_loss}')
    # utils.get_pdf(d_pred, d_obs, d_bias, model_name, note=note)
    if bias:
        utils.plot_bias(monthly_bias, plot_params, model_name, note=note)

# ---------------------------------------------------------------------------------
if __name__ == '__main__': 
    main()

