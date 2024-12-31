## train.py: training framework for proposed models
# ---------------------------------------------------------------------------------
# pytorch imports
import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group, all_gather

# std lib imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeat

# custom imports
from PrecipDataset import PrecipDataset
from Networks import *
from InceptUNet import IRNv2UNet

# ---------------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print('Missing args')
        sys.exit(20)

    model_name, mode, n_epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])  # cnn, unet, aunet, cae; train, test, val; int

    if torch.cuda.is_available():
        print('Using GPU')
        device = torch.device('cuda')
        # world_size = int(os.environ['WORLD_SIZE'])  # torch.cuda.device_count()
        # print(world_size, torch.cuda.device_count())
    else:
        print('Using CPU')
        device = torch.device('cpu')

    # TODO: DDP stuff not working >:(
    save_every = 5  # save model state every 5 epochs
    train_loader, val_loader, test_loader = prep_loaders(ddp=False)
    if mode == 'test':
        model = prep_model(model_name)
        model = load_model(model, model_name, device)
        check_accuracy(model, model_name, test_loader, device, precip=True, bias=True)
    elif mode == 'train':
        if n_epochs == 0:  # use this as hyperparam search flag
            lr = np.logspace(-5, -1, 5)
            wd = np.logspace(-3, -1, 3)
            for l in lr:
                for w in wd:
                    print(f'Learning rate: {l}, Weight decay: {w}')
                    model = prep_model(model_name)
                    optim = prep_optimizer(model, l, w)
                    trainer = Trainer(model, model_name, train_loader, val_loader, optim, device, 3)
                    trainer.train(search=True)
        elif n_epochs == 1:
            t1 = time.time()
            model = prep_model(model_name)
            optim = prep_optimizer(model)
            trainer = Trainer(model, model_name, train_loader, val_loader, optim, device, 1)
            trainer.train()
            t2 = time.time()
            print(f'1 epoch run in {t2 - t1}')
        else:
            model = prep_model(model_name)
            lr, wd = 1e-7, 1e-5  # inc: 1e-3, 1e-1  # (a)unet: 1e-3, 1e-5
            optimizer = prep_optimizer(model, lr=lr, wd=wd)
            trainer = Trainer(model, model_name, train_loader, val_loader, optimizer, device, n_epochs)
            trainer.train()
            train_params = {
                'learning_rate': lr,
                'L2_weight_decay': wd,
                'n_epochs': n_epochs
            }
            print(f'Training hyperparameters: {train_params}')
        # mp.spawn(run, args=(world_size, model_name, save_every, n_epochs), nprocs=world_size)


# ---------------------------------------------------------------------------------
def prep_model(model_name, in_channels=64):
    # create specified model
    if model_name == 'cnn':
        model = CNN(in_channels)
    elif model_name == 'unet':
        model = UNet(in_channels)
    elif model_name == 'aunet':
        model = AttentionUNet(in_channels)
    elif model_name == 'cae':
        model = CAE_LSTM(in_channels)
    elif model_name == 'inc':
        model = IRNv2UNet(in_channels)
    else:
        print(f'Model "{model_name}" not valid')

    return model

# ---------------------------------------------------------------------------------
def prep_optimizer(model, lr=1e-4, wd=1e-2):
    # might we want different optimizer 
    # what about lr? with distributed training seen debates on scaling this
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)  # use default betas?

    return optimizer

# ---------------------------------------------------------------------------------
def prep_loaders(ddp=False):
    train_ds = PrecipDataset('train')
    val_ds = PrecipDataset('val')
    test_ds = PrecipDataset('test')
    # TODO: if trying DDP again need to sort sampler out
    sampler = DistributedSampler(dataset, drop_last=True) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True, pin_memory=True, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=True, pin_memory=True, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, pin_memory=True, sampler=sampler)
    return train_loader, val_loader, test_loader

# ---------------------------------------------------------------------------------
def load_model(model, model_name, device):
    path = os.path.join(f'./models/{model_name}/params.pth')
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ---------------------------------------------------------------------------------
def check_accuracy(model, model_name, loader, device, precip=False, bias=False):
    loss_list, monthly_bias = [], []
    sel_out, sel_y, sel_t = [], [], []
    rand_months = torch.randint(len(loader) - 1, (4,)).tolist()  # select 4 random months from test set
    sel_months = [1, 4, 15, 23]

    extent = (-108, -84.25, 24, 47.75)
    lons, lats = np.arange(-108, -84, 0.25), np.arange(24, 48, 0.25)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, y, t_str) in enumerate(loader):
            print(f'Testing model on {t_str}...')
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            output = model(x)
            loss = F.mse_loss(output, y)  # this should be one value as mean mse for all days in the month
            loss_list.append(loss.detach().cpu())

            if bias:
                b = calc_daily_bias(output, y)
                monthly_bias.append(b)
            if precip and i in sel_months:
                sel_out.append(output)
                sel_y.append(y)
                sel_t.append(t_str)

        mean_loss = np.mean(loss_list)
        print(f'Mean MSE loss on test set: {mean_loss}')
        if precip:
            plot_precip(sel_out, sel_y, sel_t, extent, (lons, lats), model_name)
        elif bias:
            plot_bias(monthly_bias, extent, (lons, lats), model_name)

# ---------------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, model_name, train_loader, val_loader, optimizer, device, n_epochs, save_every=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        # TODO: supply this to __init__ instead
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-4, epochs=n_epochs, steps_per_epoch=len(train_loader))
        self.device = device
        self.n_epochs = n_epochs
        self.save_every = save_every
        self.model_path = f'./models/{model_name}/'
        self.train_llist = []
        self.val_llist = []
        self._ctr = 0

    def _best_loss(self):
        print(self.train_llist)
        print(self.val_llist)
        print(f'min train loss: {np.amin(self.train_llist)} @ epoch {np.argmin(self.train_llist) + 1}')
        print(f'min val loss: {np.amin(self.val_llist)} @ epoch {np.argmin(self.val_llist) + 1}')

    def _plot_loss(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(len(self.train_llist)), self.train_llist, label='train loss')
        ax.plot(np.arange(len(self.val_llist)), self.val_llist, label='val loss')
        ax.set(xlabel='epoch', ylabel='loss', ylim=(0, 2 * np.amax(self.train_llist)))
        plt.legend()
        fig.tight_layout()
        sv_pth = os.path.join(self.model_path, 'loss_dpr.png')
        plt.savefig(sv_pth, dpi=300.0)

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        out = self.model(source)
        # how can we penalize negative predictions?
        # this from https://stackoverflow.com/questions/50711530/what-would-be-a-good-loss-function-to-penalize-the-magnitude-and-sign-difference
        # switching these back to mean reduction?
        loss = F.mse_loss(out, target, reduction='mean') + torch.mean(F.relu(-1 * out))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_val_batch(self, source, target):
        out = self.model(source)
        val_loss = F.mse_loss(out, target, reduction='mean') + torch.mean(F.relu(-1 * out))
        return val_loss.cpu().detach().numpy()

    def _run_epoch(self, epoch, save):
        e_loss = []
        self.model.train()
        for i, (source, target, t_str) in enumerate(self.train_loader):
            # TODO: either stick with removing whole month 2020-12 or strip last day of each?
            if source.shape[0] != target.shape[0]:
                # print(f'Skipping {t_str}, source/target shapes do not match!')
                continue
            source = source.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.float32)
            train_loss = self._run_batch(source, target)
            e_loss.append(train_loss)
            #print(self.scheduler.get_last_lr())
            self.scheduler.step()  # for OneCycle scheduler
        
        e_val_loss = []
        self.model.eval()
        with torch.no_grad():
            for j, (s, t, t_str) in enumerate(self.val_loader):
                if s.shape[0] != t.shape[0]:
                    # print(f'Skipping {t_str}, source/target shapes do not match!')
                    continue
                s = s.to(self.device, dtype=torch.float32)
                t = t.to(self.device, dtype=torch.float32)
                val_loss = self._run_val_batch(s, t)
                e_val_loss.append(val_loss)

        self.train_llist.append(np.mean(e_loss))
        self.val_llist.append(np.mean(e_val_loss))

    def _save_point(self, epoch):
        state = self.model.state_dict()
        sv_pth = os.path.join(self.model_path, 'params_dpr.pth')
        torch.save(state, sv_pth)
        print(f'Model state saved at epoch {epoch}')

    @property
    def _stop_early(self):
        # use val instead? really volatile is the problem
        if self.train_llist[-1] >= self.train_llist[-2]:
            self._ctr += 1
        elif self.train_llist[-1] < self.train_llist[-2]:
            self._ctr = 0

        if self._ctr > 10:
            return True
        else:
            return False

    def train(self, search=False):
        for epoch in range(1, self.n_epochs + 1):
            save = (epoch % self.save_every == 0)
            self._run_epoch(epoch, save)
            t_loss = self.train_llist[-1]
            v_loss = self.val_llist[-1]
            print(f'Epoch {epoch} | Train loss: {t_loss}, Val loss: {v_loss}')
            print()
            
            if v_loss == np.min(self.val_llist):
                self._save_point(epoch)
            #self.scheduler.step()
            if epoch > 1:
                if self._stop_early:
                    print(f'No improvement for 10 epochs, stopping now at epoch {epoch}')
                    break
        print('Run completed.')

        if not search:
            self._plot_loss()
        else:
            self._best_loss()

# ---------------------------------------------------------------------------------
if __name__ == '__main__': 
    main()

