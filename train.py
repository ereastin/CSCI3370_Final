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
        check_accuracy(model, model_name, test_loader, device)
    elif mode == 'train':
        if n_epochs == 0:  # use this as hyperparam search flag
            lr = np.logspace(-5, -1, 6)
            wd = np.logspace(-5, -1, 3)
            for l in lr:
                for w in wd:
                    print(f'Learning rate: {l}, Weight decay: {w}')
                    model = prep_model(model_name)
                    optim = prep_optimizer(model, l, w)
                    trainer = Trainer(model, model_name, train_loader, val_loader, optim, device)
                    trainer.train(3, search=True)
        elif n_epochs == 1:
            t1 = time.time()
            model = prep_model(model_name)
            optim = prep_optimizer(model)
            trainer = Trainer(model, model_name, train_loader, val_loader, optim, device)
            trainer.train(1)
            t2 = time.time()
            print(f'1 epoch run in {t2 - t1}')
        else:
            model = prep_model(model_name)
            lr, wd = 1e-3, 1e-1  # inc: 1e-3, 1e-1  # (a)unet: 1e-3, 1e-5
            optimizer = prep_optimizer(model, lr=lr, wd=wd)
            trainer = Trainer(model, model_name, train_loader, val_loader, optimizer, device)
            trainer.train(n_epochs)
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
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

# ---------------------------------------------------------------------------------
def pcc(a, b):
    a_mn = np.mean(a)
    b_mn = np.mean(b)
    return np.sum((a - a_mn) * (b - b_mn)) / np.sqrt(np.sum((a - a_mn) ** 2) * np.sum((b - b_mn) ** 2))

# ---------------------------------------------------------------------------------
def check_accuracy(model, model_name, loader, device):
    model = model.to(device)
    loss_list = []
    rand_months = torch.randint(len(loader) - 1, (4,)).tolist()  # select 4 random months from test set
    
    lvls = np.linspace(0, 32, 17)
    extent = (-108, -84.25, 24, 47.75)
    lons, lats = np.arange(-108, -84, 0.25), np.arange(24, 48, 0.25)
    #cmap = 'Blues'
    clrs = ['#FFFFFF','#BEFFFF','#79C8FF','#3E62FF','#2F2DDE','#79DA62','#58D248','#3BBF3D','#28A83A','#F8FB64', '#FFD666','#FFA255','#FF6039','#F61F1F','#CD3B3B','#AC3333','#CD1599','#C725E0']
    cmap = colors.LinearSegmentedColormap.from_list('p_cmap', clrs, 18)
    cmap.set_bad(color='white')
    cmap.set_under(color='white')
    
    model.eval()
    with torch.no_grad():
        for i, (x, y, t_str) in enumerate(loader):
            print(f'Testing model on {t_str}...')
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU -> may need to specify dtype here..?
            y = y.to(device=device, dtype=torch.float32)
            output = model(x)
            # recall these are values for one month of data!
            loss = F.mse_loss(output, y)  # this should be one value as mean mse for all days in the month..
            loss_list.append(loss.detach().cpu())
            # rmse = torch.sqrt(mse)

            if i in rand_months:
                sel_dates = torch.randint(output.shape[0], (3,)).tolist()
                out_sel, y_sel = output[sel_dates].squeeze(1), y[sel_dates].squeeze(1)
                # so these are now (B, 1, 96, 96) -> (len(sel_dates), 96, 96)
                outs = list(torch.split(out_sel, 1, dim=0))
                ys = list(torch.split(y_sel, 1, dim=0))
                outs = [im.squeeze(0).numpy(force=True) for im in outs]
                ys = [im.squeeze(0).numpy(force=True) for im in ys]
                datas, rmses, pccs = [], [], []
                for j in range(len(outs)):
                    datas.append(outs[j])
                    datas.append(ys[j])
                    rmses.append(rmse(outs[j], ys[j]))
                    pccs.append(pcc(outs[j], ys[j]))
                
                fig, axs = plt.subplots(3, 2, subplot_kw={'projection': ccrs.PlateCarree()})
                axs = axs.flatten()
                fig.suptitle(f'Predicted vs. Observed Data on {t_str}: {model_name}')
                for i, (ax, data) in enumerate(zip(axs, datas)):
                    cs = ax.contourf(lons, lats, data, lvls, transform=ccrs.PlateCarree(), cmap=cmap, extend='max')
                    ax.set_extent(extent, crs=ccrs.Geodetic())
                    ax.add_feature(cfeat.STATES)
                    if i == 0:
                        ax.set_title(f'Predicted, RMSE: {rmses[0]:.3f}, PCC: {pccs[0]:.3f}', fontsize=6)
                    if i == 1: 
                        ax.set_title('Observed', fontsize=6)
                    if i % 2 == 0:
                        ax.set_title(f'RMSE: {rmses[i // 2]:.3f}, PCC: {pccs[i // 2]:.3f}', fontsize=6)

                fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.1, label='Accumulated Precip [mm]')
                img_fname = f'{t_str}_sel.png'
                img_path = os.path.join(f'./models/{model_name}/', img_fname)
                plt.savefig(img_path, dpi=300, bbox_inches='tight')

        mean_loss = np.mean(loss_list)
        print(f'Mean MSE loss on test set: {mean_loss}')
    return

# ---------------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, model_name, train_loader, val_loader, optimizer, device, save_every=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [5, 100, 130], gamma=0.5)
        self.device = device
        self.save_every = save_every
        self.model_path = f'./models/{model_name}/'
        self.train_llist = []
        self.val_llist = []
        self._ctr = 0

    def _best_loss(self):
        print(f'min train loss: {np.amin(self.train_llist)} @ epoch {np.argmin(self.train_llist)}')
        print(f'min val loss: {np.amin(self.val_llist)} @ epoch {np.argmin(self.val_llist)}')

    def _plot_loss(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(len(self.train_llist)), self.train_llist, label='train loss')
        ax.plot(np.arange(len(self.val_llist)), self.val_llist, label='val loss')
        ax.set(xlabel='epoch', ylabel='loss', ylim=(0, 1.1 * np.amax(self.train_llist)))
        plt.legend()
        fig.tight_layout()
        sv_pth = os.path.join(self.model_path, 'loss.png')
        plt.savefig(sv_pth, dpi=300.0)

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        out = self.model(source)
        # how can we penalize negative predictions?
        # this from https://stackoverflow.com/questions/50711530/what-would-be-a-good-loss-function-to-penalize-the-magnitude-and-sign-difference
        loss = F.mse_loss(out, target, reduction='sum') + torch.sum(F.relu(-1 * out))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_val_batch(self, source, target):
        out = self.model(source)
        val_loss = F.mse_loss(out, target, reduction='sum') + torch.sum(F.relu(-1 * out))
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
        sv_pth = os.path.join(self.model_path, 'params.pth')
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

    def train(self, max_epochs, search=False):
        for epoch in range(1, max_epochs + 1):
            save = (epoch % self.save_every == 0)
            self._run_epoch(epoch, save)
            t_loss = self.train_llist[-1]
            v_loss = self.val_llist[-1]

            if v_loss == np.min(self.val_llist):
                self._save_point(epoch)
                print(f'Epoch {epoch} | Train loss: {t_loss}, Val loss: {v_loss}')
                print()
            self.scheduler.step()
            if epoch > 1:
                if self._stop_early:
                    print(f'No improvement for 10 epochs, stopping now at epoch {epoch}')
                    break
        print('Run completed.')

        if not search:
            self._plot_loss()
        else:
            self._best_loss()

## DDP stuff
# ---------------------------------------------------------------------------------
def run(rank, world_size, model_name, save_every, total_epochs):
    #try:
    ddp_setup(rank, world_size)
    print(f'Rank {rank} setup')
    model, optimizer = prep_objects(model_name, 64)  # worth not hard coding this?
    print(f'Rank {rank} model, optimizer setup')
    loader = prep_loader()
    print(f'Rank {rank} loader setup')
    trainer = Trainer(model, model_name, loader, optimizer, rank, world_size, save_every)
    print(f'Rank {rank} trainer setup')
    trainer.train(total_epochs)
    destroy_process_group()
    #except Exception as e:
    #    print(f'GPU {rank} encountered error {e}')
    #    destroy_process_group()

# ---------------------------------------------------------------------------------
def ddp_setup(rank, world_size):
    """
    rank (int): unique ID for each process
    world_size (int): total number of processes
    """
    # os.environ['MASTER_ADDR'] = 'g009'  # is this right.. check slurm thing from torch examples
    os.environ['MASTER_PORT'] = '23345'
    # print(f'Rank {rank} setting device')
    torch.cuda.set_device(rank)
    # print(f'Rank {rank} init_process_group')
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

# ---------------------------------------------------------------------------------
class DDPTrainer:
    def __init__(self, model, model_name, loader, optimizer, gpu_id, world_size, save_every):
        self.gpu_id = gpu_id
        # this the right place for this? why doesn't DDP wrap self.model?
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model.to(gpu_id))
        self.model = DDP(model, device_ids=[gpu_id])
        self.loader = loader
        self.optimizer = optimizer
        self.world_size = world_size
        self.save_every = save_every
        self.model_path = f'./models/{model_name}/'
        self.loss_list = []

    def _plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(self.loss_list)), self.loss_list, label='train loss')
        ax.set(xlabel='epoch', ylabel='loss')
        plt.legend()
        sv_pth = os.path.join(self.model_path, 'loss.png')
        plt.savefig(sv_pth, dpi=300.0)

    def _run_batch(self, source, target, collect_loss=False):
        self.optimizer.zero_grad()
        out = self.model(source)
        loss = F.mse_loss(out, target)
        # would be careful about where collect_loss is
        # before .backward() vs. before optimizer.step() maybe different?
        # sounds like should be before optimizer.step()?
        loss.backward()
        
        # collect loss for plotting (?) need loss.item() for these?
        if collect_loss:
            all_loss = [torch.zeros_like(loss) for _ in range(self.world_size)]
            all_gather(all_loss, loss)
            if self.gpu_id == 0:
                print(f'Collecting loss...')
                self.loss_list.append(torch.mean(torch.stack(all_loss)).cpu())
        
        self.optimizer.step()

    #TODO: add _run_val_batch to get validation loss at each epoch?

    def _run_epoch(self, epoch, save):
        # not sure if this is needed, something re: shuffling if we care
        # self.loader.sampler.set_epoch(epoch)
        print(f'[GPU {self.gpu_id}] Epoch {epoch} | Steps: {len(self.loader)}')  # does this split it up auto?
        for i, (source, target, t_str) in enumerate(self.loader):
            # TODO: either stick with removing whole month 2020-12 or strip last day of each?
            if source.shape[0] != target.shape[0]:
                print(f'Skipping {t_str}, source/target shapes do not match!')
                continue
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            collect_loss = (i == (len(self.loader) - 1))
            self._run_batch(source, target, collect_loss)

    def _save_point(self, epoch):
        state = self.model.module.state_dict()
        sv_pth = os.path.join(self.model_path, 'params.pth')
        torch.save(state, sv_pth)
        print(f'Model state saved at epoch {epoch}')

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            save = ((epoch + 1) % self.save_every == 0)
            self._run_epoch(epoch, save)
            if self.gpu_id == 0 and save:
                self._save_point(epoch)
                print(f'Epoch {epoch} | MSE loss: {self.loss_list[-1]}')
                print()
        print('Run completed.')
        self._plot_loss()

# ---------------------------------------------------------------------------------
if __name__ == '__main__': 
    main()

