import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from socket import gethostname
import numpy as np
from datetime import timedelta
from argparse import ArgumentParser
import os
import sys
sys.path.append('/home/eastinev/AI')
import time
## personal imports
from TrainHelper import TrainHelper
from InceptUNet3D import IRNv4_3DUNet
from Incept3D import IRNv4_3D
from InceptUNet import IRNv4UNet
from v2 import UNet, MultiUNet
from simple import Simple
from simple2d import Simple2
from Dummy import Dummy
from PrecipDataset import PrecipDataset
from OTPrecipDataset import OTPrecipDataset
from decorators import timeit
from perceptual import SqueezeNet

# for Lazy module dry-runs.. handle this better for other input shapes
C, D, H, W = 6, 28, 81, 145 #6, 16, 80, 144
FROM_LOAD = False
MIN, MAX = np.log(1.1), np.log(101)
RET_AS_TNSR = True

# =================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('n_epochs', type=int)
    parser.add_argument('-s', '--search', action='store_true')
    parser.add_argument('-d', '--ddp', action='store_true')
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    parser.add_argument('-t', '--tag', type=str, default='')
    args = parser.parse_args()

    model_name, n_epochs, ddp = args.model_name, args.n_epochs, args.ddp
    exp, season = args.exp, args.season
    tag = args.tag
    search = args.search
    model_id_tag = season + tag
    weekly = True  # check memory on this vs monthly.. max ~3GB per month so why fail with 16GB per cpu?

    print(f'Job ID: {os.environ["SLURM_JOBID"]}', flush=True)

    # Set up DDP-necessary identifications
    if ddp:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
        assert gpus_per_node == torch.cuda.device_count()
        print(f'Rank {rank} of {world_size} on {gethostname()} with {gpus_per_node} allocated GPUs per node', flush=True)

        dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=2))
        if rank == 0: print(f'Init: {dist.is_initialized()}', flush=True)
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0
        local_rank = torch.device('cuda')

    if search:  # random search instead of grid search
        base = np.random.choice([6, 8, 12, 16, 24, 32])
        lin_act = np.random.choice(np.append(np.linspace(0.05, 0.3, 10), np.array([1.0])))
        Na = np.random.choice(np.arange(1, 6))
        Nb, Nc = 2 * Na, Na
        # just force to 1 each
        # Na, Nb, Nc = 1, 1, 1
        lr = np.random.choice(np.logspace(-5, -1, 20))
        max_lr = lr * 50
        wd = np.random.choice(np.append(np.logspace(-3, -1, 10), np.linspace(0, 0.5, 5)))
        drop_p = np.random.choice(np.linspace(0.0, 0.3, 10))
        bias = np.random.choice(np.array([True, False]))
        opt_type = 'adamw' #np.random.choice(np.array(['adamw', 'adam', 'sgd']))
        loss_fn = np.random.choice(np.array([huber, mse, wt_mse, mae]))
        hps = {
            'base': base, 'lin_act': lin_act, 'Na': Na, 'Nb': Nb, 'Nc': Nc, 'loss_fn': loss_fn.__name__,
            'optim': opt_type, 'lr': lr, 'max_lr': max_lr, 'wd': wd, 'drop_p': drop_p, 'bias': bias
        }
    else:
        base = 36
        lin_act = .1
        Na, Nb, Nc = 5, 10, 5
        lr = 1e-4
        wd = 0.15
        drop_p = 0#.05  # probably just leave as 0 these dont do great with CNNs?
        bias = True
        opt_type = 'adamw'
        loss_fn = SqueezeNet().to(local_rank).float()
        #loss_fn = comp_loss_fn
        hps = {
            'base': base, 'lin_act': lin_act, 'Na': Na, 'Nb': Nb, 'Nc': Nc, 'loss_fn': 'mse',
            'optim': opt_type, 'lr': lr, 'wd': wd, 'drop_p': drop_p, 'bias': bias
        }

    #model = UNet(depth=D, init_c=32, dim=3, bias=bias).to(local_rank).float()
    #model = MultiUNet(n_vars=C, depth=D, spatial_dim=2, init_c=128, embedding_dim=32, bias=bias).to(local_rank).float()
    model = Simple(C, depth=D, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act).to(local_rank).float()
    #model = Simple2(C * D, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act).to(local_rank).float()
    #model = IRNv4_3DUNet(C, depth=D, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act).to(local_rank).float()
    # model = Dummy().to(local_rank).float()

    if FROM_LOAD:
        model = load_model(model, model_name, model_id_tag, local_rank)
    else:
        # dry-run for Lazy modules -- must be called before DDP init
        model(torch.ones(1, C, D, H, W).to(local_rank))

    # Convert model to DDP-accessible type for distributed training
    if ddp: model = DDP(model, device_ids=[local_rank])

    # Create optimizer
    optimizer = prep_optimizer(model.parameters(), lr, wd, opt_type)

    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, cooldown=0)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 350], gamma=0.5)

    # Create DataLoaders
    train_loader, val_loader, sampler = prep_loaders(exp, season, rank, world_size, weekly=weekly, ddp=ddp)
    train_loader_len = len(train_loader.dataset) + (len(train_loader.dataset) % world_size) if ddp else len(train_loader.dataset)
 
    # Create TrainHelper to manage training progress
    if rank == 0: helper = TrainHelper(model_name, tag=model_id_tag, hyperparams=hps)

    # Create CPU backend for aggregating losses across GPUs
    if ddp: cpu_grp = dist.new_group(backend='gloo')
    if ddp: dist.barrier()

    # Create tensor for collecting stop training signal across GPUs
    stop_signal = torch.tensor([0], dtype=torch.int)

    # Training loop
    try:
        curr_lr = lr
        if rank == 0: t1 = time.time()
        for epoch in range(1, n_epochs + 1):
            if rank == 0: tx = time.time()
            if ddp: sampler.set_epoch(epoch)  # necessary for DataLoaders w/DDP to correctly shuffle

            # Train epoch
            #l = train(model, local_rank, train_loader, optimizer, loss_fn)
            l, n = train(model, local_rank, train_loader, optimizer, loss_fn)

            # Collect training loss and validate
            if ddp:
                dist.barrier()
                all_loss = [torch.zeros_like(l, device=torch.device('cpu')) for _ in range(world_size)]
                all_n = [torch.zeros_like(n, device=torch.device('cpu')) for _ in range(world_size)]
                dist.all_gather(all_loss, l, group=cpu_grp)
                dist.all_gather(all_n, n, group=cpu_grp)
                val_loss = torch.tensor([0.0], device=torch.device('cpu'))
                if rank == 0:
                    #train_loss = torch.sum(torch.tensor(all_loss)) / train_loader_len
                    train_loss = torch.sum(torch.tensor(all_loss)) / torch.sum(torch.tensor(all_n))
                    if train_loss is torch.nan: raise ValueError  # TODO: this doesn't actually work?
                    val_loss = validate(model, local_rank, val_loader, loss_fn, ddp=ddp)
                dist.barrier()
                dist.broadcast(val_loss, src=0, group=cpu_grp)
            else:
                train_loss = l / n
                if train_loss is torch.nan: raise ValueError
                val_loss = validate(model, local_rank, val_loader, loss_fn, ddp=ddp)
                #val_loss = 0

            # Step LR scheduler
            #scheduler.step()
            scheduler.step(val_loss)
            tmp_lr = scheduler.get_last_lr()[0]
            if tmp_lr != curr_lr:
                print(f'Learning rate update on plateau at epoch {epoch}: {curr_lr} -> {tmp_lr}', flush=True)
                curr_lr = tmp_lr

            # Checkpoint model
            if ddp: dist.barrier()  # make sure all training in epoch is done before saving
            if rank == 0:
                helper.add(train_loss, val_loss)
                helper.print(epoch)
                if ddp:
                    stop_signal = helper.checkpoint(model.module, epoch)
                else:
                    stop_signal = helper.checkpoint(model, epoch, search)

            # Check for stop training signal
            if ddp: dist.barrier()  # make sure to not move forward with training while saving in rank0
            if ddp: dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM, group=cpu_grp)
            if stop_signal.item() > 0:
                print(f'No improvement, stopping at epoch {epoch}', flush=True)
                break

            if rank == 0:
                ty = time.time()
                print(f'Epoch {epoch} done in {ty - tx} seconds', flush=True)

        if rank == 0:
            t2 = time.time()
            print(f'Run completed in {t2 - t1}', flush=True)
        #helper._save_point(model, epoch)

    except ValueError as e:
        print(f'[ERROR]: Loss is nan @ epoch {epoch} with message {e}, exiting...')
    except Exception as e:
        print(f'[ERROR]: Main on {local_rank} @ {time.time()} -- {e}')
        raise e
    finally:
        if rank == 0: helper.finish(search)
        cleanup(ddp)

# =================================================================================
#@timeit
def train(model, device, train_loader, optimizer, loss_fn):
    try:
        train_loss = 0
        losses, Ns = [], []
        model.train()
        for i, (source, target, tt) in enumerate(train_loader):
            if not RET_AS_TNSR:
                # compute
                source = source.to_dataarray().to_numpy()
                source = torch.tensor(source).permute(1, 0, 2, 3, 4) # return as (time, var, (lev), lat, lon) 
                target = target.to_dataarray().to_numpy()
                target = torch.tensor(target).permute(1, 0, 2, 3) # return as (time, var, lat, lon) 

            source, target = source.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            out = model(source)
            loss = loss_fn(out, target)
            #train_loss += loss.item()
            losses.append(loss.item())
            Ns.append(source.shape[0])
            loss.backward()
            optimizer.step()
        return torch.sum(torch.tensor(losses) * torch.tensor(Ns)), torch.sum(torch.tensor(Ns)) 
        #return torch.tensor(train_loss)
    except Exception as e:
        print(f'[ERROR]: Training on {device} @ {time.time()} -- {e}', flush=True)
        raise e
        return

# ---------------------------------------------------------------------------------
def validate(model, device, val_loader, loss_fn, ddp=False):
    try:
        losses, Ns = [], []
        model.eval()
        val_model = model if not ddp else model.module
        with torch.no_grad():
            for i, (source, target, tt) in enumerate(val_loader):
                source, target = source.to(device).float(), target.to(device).float()
                out = val_model(source)
                loss = loss_fn(out, target)
                losses.append(loss.item())
                Ns.append(source.shape[0])

        val_loss = np.sum(np.array(Ns) * np.array(losses)) / np.sum(np.array(Ns))
        return torch.tensor(val_loss)
    except Exception as e:
        print(f'[ERROR]: Validating on {device} @ {time.time()} -- {e}', flush=True)
        raise e
        return

# ---------------------------------------------------------------------------------
def huber(pred, target):
    return F.huber_loss(pred, target, delta=3)

def mse(pred, target):
    loss = F.mse_loss(pred, target)
    return loss

def wt_mse(pred, target):
    wt = torch.abs(target)
    return F.mse_loss(pred, target, weight=wt)

def mae(pred, target):
    return F.l1_loss(pred, target)

def wt_mae(pred, target):
    wt = torch.where(target <= MIN, MIN, target)
    wt = torch.where(wt >= MAX, MAX, wt)
    l1 = torch.mean(wt * torch.abs(pred - target))
    return l1

def pcc(pred, target):
    t = torch.exp(target) - 1
    mt = torch.mean(t, dim=(2, 3), keepdim=True)
    ts = t - mt
    p = torch.exp(pred) - 1
    mp = torch.mean(p, dim=(2, 3), keepdim=True)
    ps = p - mp
    eps = 0
    pcc_loss = torch.sum(ps * ts) / torch.sqrt(torch.sum(ps ** 2) * torch.sum(ts ** 2) + eps)
    return 1 - pcc_loss

def fft(pred, target):
    p = pred - torch.mean(pred, dim=(2, 3), keepdim=True)
    t = target - torch.mean(target, dim=(2, 3), keepdim=True)
    fft_pred = torch.abs(torch.fft.fft2(p, norm='ortho'))# - torch.mean(p, dim=(2, 3), keepdim=True), norm='ortho'))
    fft_target = torch.abs(torch.fft.fft2(t, norm='ortho'))# - torch.mean(t, dim=(2, 3), keepdim=True), norm='ortho'))
    fft_wt = torch.log(1 + F.l1_loss(fft_pred, fft_target, reduction='none'))
    fft_loss = F.l1_loss(fft_pred, fft_target, weight=fft_wt)
    return fft_loss

def cross_entropy(pred, target):
    target = target.to(torch.long)
    #wt = torch.tensor([1/.99, 1/.01, 0, 0, 0]).cuda()
    loss = F.cross_entropy(pred, target)#, weight=wt)
    #print(loss.item())
    return loss

def dice(pred, target):
    a = pred.contiguous().view(-1)
    b = target.contiguous().view(-1)
    return 1 - (2 * (a * b).sum() + 1e-5) / (a.sum() + b.sum() + 1e-5)
    
def comp_loss_fn(pred, target):
    mse_loss = mse(pred, target)
    #mae_loss = mae(pred, target)
    fft_loss = fft(pred, target)
    #return mse_loss + 0.25 * fft_loss
    return mse_loss + 0.35 * fft_loss

# ---------------------------------------------------------------------------------
def prep_model(model_name, tag, in_channels=64):
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
        case 'unet':
            return UNet(16)
        case _:
            print(f'Model "{model_name}" not valid')
            sys.exit(-21)

# ---------------------------------------------------------------------------------
def load_model(model, model_name, tag, device):
    path = os.path.join(f'./models/{model_name}/params_{tag}.pth')
    model.load_state_dict(torch.load(path, map_location=device))

    return model

# ---------------------------------------------------------------------------------
def prep_optimizer(model_params, lr=1e-4, wd=1e-2, opt_type='adamw'):
    match opt_type:
        case 'adamw':
            optimizer = optim.AdamW(model_params, lr=lr, weight_decay=wd)  # use default betas?
        case 'adam':
            optimizer = optim.Adam(model_params, lr=lr, weight_decay=wd)
        case 'sgd':
            optimizer = optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=0.9)

    return optimizer

# ---------------------------------------------------------------------------------
def prep_loaders(exp, season, rank, world_size, weekly=False, ddp=False):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    prefetch = 1
    shuffle = True
    train_ds = OTPrecipDataset('train', exp, season, weekly=weekly, shuffle=shuffle, ret_as_tnsr=RET_AS_TNSR)
    val_ds = OTPrecipDataset('val', exp, season, weekly=weekly, ret_as_tnsr=RET_AS_TNSR)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if ddp else None

    if sampler:
        train_loader = DataLoader(train_ds, batch_size=None, sampler=sampler, num_workers=n_workers, prefetch_factor=prefetch)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=None,
            shuffle=shuffle,
            num_workers=n_workers,
            prefetch_factor=prefetch,
            persistent_workers=True
        )

    val_loader = DataLoader(val_ds, batch_size=None, num_workers=n_workers, prefetch_factor=prefetch)

    return train_loader, val_loader, sampler

# ---------------------------------------------------------------------------------
def cleanup(ddp):
    if ddp:
        dist.destroy_process_group()
    else:
        pass  # anything we need here?

# =================================================================================
if __name__ == '__main__':
    main()

