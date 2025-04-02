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

# custom imports
from PrecipDataset import PrecipDataset
from Networks import *
from InceptUNet import IRNv4UNet
from InceptUNet3D import IRNv4_3DUNet

sys.path.append('/home/eastinev/AI')
import analysis_utils as utils
import paths as pth

# ---------------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('-r', '--rmvar', type=str)
    parser.add_argument('-z', '--zero', action='store_true')
    parser.add_argument('-p', '--precip', action='store_true')
    parser.add_argument('-b', '--bias', action='store_true')
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    rm_var, zero = args.rmvar, args.zero
    exp, season = args.exp, args.season
    # TODO: what about removing vertical shear? set all to..? average of midlevel?
    note = season + '_' + 'ctrl' if rm_var is None else f'{"no" if zero else "mn"}{rm_var}'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    test_loader = prep_loader(exp, season, rm_var, zero)
    model = prep_model(model_name, in_channels=6)
    model = load_model(model, model_name, season, device)

    print(f'Running test experiment: {note}')
    check_accuracy(model, model_name, test_loader, device, note=note, precip=args.precip, bias=args.bias)

# ---------------------------------------------------------------------------------
def prep_model(model_name, in_channels=64):
    # create specified model
    match model_name:
        case 'inc':
            return IRNv4UNet(in_channels)
        case 'inc3d':
            return IRNv4_3DUNet(in_channels, depth=35, Na=1, Nb=2, Nc=1, drop_p=0.3)
        case _:
            print(f'Model "{model_name}" not valid')
            sys.exit(-21)

# ---------------------------------------------------------------------------------
def prep_loader(exp, season, rm_var, zero, weekly=False):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    test_ds = PrecipDataset('test', exp, season, weekly, rm_var, zero)
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
    sel_out, sel_y, sel_t = [], [], []
    d_pred, d_obs, d_bias = [], [], []
    rand_months = torch.randint(len(loader) - 1, (4,)).tolist()  # select 4 random months from test set
    sel_months = rand_months # [1, 4, 15, 23]

    extent = (-140, -50.625, 14, 53.5)
    dlon, dlat = 0.625, 0.5
    lons, lats = np.arange(-140, -50.625 + dlon, dlon), np.arange(14, 53.5 + dlat, dlat)
    # for zhang_repr
    # extent = (-108, -84.25, 24, 47.75)
    # lons, lats = np.arange(-108, -84, 0.25), np.arange(24, 48, 0.25)
    plot_params = {'extent': extent, 'lons': lons, 'lats': lats}

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, y, t_str) in enumerate(loader):
            print(f'Testing model on {t_str}...')
            x, y = x.to(device=device), y.to(device=device)
            output = model(x)
            loss = F.mse_loss(output, y) # + torch.mean(F.relu(-1 * output))  # include relu in MSE calcs..?
            loss_list.append(loss.item())

            # get pdf stuff
            # pred, obs, bb = utils.calc_daily_accum(output, y)
            # d_pred.append(pred)
            # d_obs.append(obs)
            # d_bias.append(bb)

            if bias:
                b = utils.calc_daily_bias(output, y, steps_per_day=8)  # (B // 4, 1, 96, 96), B // 4 being number of days
                monthly_bias.append(b)
            if precip and i in sel_months:
                sel_out.append(output)
                sel_y.append(y)
                sel_t.append(t_str)

    mean_loss = np.mean(loss_list)
    print(f'Mean MSE loss on test set: {mean_loss}')
    # utils.get_pdf(d_pred, d_obs, d_bias, model_name, note=note)
    if precip:
        utils.plot_precip(sel_out, sel_y, sel_t, plot_params, model_name, note=note)
    if bias:
        utils.plot_bias(monthly_bias, plot_params, model_name, note=note)

# ---------------------------------------------------------------------------------
if __name__ == '__main__': 
    main()

