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
from TrainHelper import TrainHelper
from InceptUNet3D import IRNv4_3DUNet
from Incept3D import IRNv4_3D
from InceptUNet import IRNv4UNet
from PrecipDataset import PrecipDataset

# for Lazy module dry-runs.. handle this better for other input shapes
C, D, H, W = 6, 35, 80, 144

# =================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('n_epochs', type=int)
    parser.add_argument('-r', '--rmvar', type=str)
    parser.add_argument('-z', '--zero', action='store_true')
    parser.add_argument('-s', '--search', action='store_true')
    parser.add_argument('-d', '--ddp', action='store_true')
    parser.add_argument('-op', '--optimparams', type=str)
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    args = parser.parse_args()

    model_name, n_epochs, ddp = args.model_name, args.n_epochs, args.ddp
    rm_var, zero = args.rmvar, args.zero
    exp, season = args.exp, args.season
    search = args.search
    if args.optimparams is not None:
        op = args.optimparams.split(':')
        lr, wd = float(op[0]), float(op[1])
    # TODO: what about removing vertical shear? set all to..? average of midlevel?
    note = 'ctrl' if rm_var is None else f'{"no" if zero else "mn"}{rm_var}'
    weekly = True
    print(f'Job ID: {os.environ["SLURM_JOBID"]}')

    if ddp:
        # TODO: does this work as its own function?
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
        assert gpus_per_node == torch.cuda.device_count()
        print(f'Rank {rank} of {world_size} on {gethostname()} which has {gpus_per_node} allocated GPUs per node', flush=True)

        dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=2))
        if rank == 0: print(f'Init?: {dist.is_initialized()}', flush=True)
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0  # this so we can use the conditionals below
        local_rank = torch.device('cuda')

    if search:  # random search instead of grid search
        base = np.random.choice([8, 12, 16, 24, 32])
        lin_act = np.random.choice(np.linspace(0.01, 0.3, 20))
        #Na = np.random.choice(np.arange(1, 6))
        #Nb, Nc = 2 * Na, Na
        # just force to 1 each
        Na, Nb, Nc = 1, 1, 1
        lr = np.random.choice(np.logspace(-5, -1, 20))
        wd = np.random.choice(np.append(np.logspace(-3, -1, 10), np.linspace(0, 0.5, 5)))
        drop_p = np.random.choice(np.linspace(0.1, 0.3, 10))
        bias = np.random.choice(np.array([True, False]))
        hps = {
            'base': base, 'lin_act': lin_act, 'Na': Na, 'Nb': Nb, 'Nc': Nc,
            'lr': lr, 'wd': wd, 'drop_p': drop_p, 'bias': bias
        }
    else:
        '''
        Run completed in 2079.592135667801
        Hyperparameters: 
        {'base': np.int64(16), 'lin_act': np.float64(0.2694736842105263), 'Na': 1, 'Nb': 1, 'Nc': 1, 'lr': np.float64(0.00011288378916846884), 'wd': np.float64(0.007742636826811269), 'drop_p': np.float64(0.2333333333333333), 'bias': np.False_}
        Min train loss: 24.324016571044922 @ epoch 5
        Min val loss: 22.316369516981972 @ epoch 5
        Run completed in 2152.9099204540253
        Hyperparameters: 
        {'base': np.int64(12), 'lin_act': np.float64(0.05578947368421053), 'Na': 1, 'Nb': 1, 'Nc': 1, 'lr': np.float64(0.00018329807108324357), 'wd': np.float64(0.001), 'drop_p': np.float64(0.1), 'bias': np.False_}
        Min train loss: 22.34442138671875 @ epoch 5
        Min val loss: 21.096297783984078 @ epoch 5
        '''
        base = 16
        lin_act = 0.269
        Na, Nb, Nc = 1, 1, 1
        lr = 4 * 1.1e-4
        wd = 0.0077
        drop_p = 0.233
        bias = False
        hps = {
            'base': base, 'lin_act': lin_act, 'Na': Na, 'Nb': Nb, 'Nc': Nc,
            'lr': lr, 'wd': wd, 'drop_p': drop_p, 'bias': bias
        }

    model = IRNv4_3DUNet(6, depth=35, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act).to(local_rank).float()

    # dry-run for Lazy modules -- must be called before DDP init
    model(torch.ones(1, C, D, H, W).to(local_rank))

    if ddp: model = DDP(model, device_ids=[local_rank])

    optimizer = prep_optimizer(model.parameters(), lr, wd)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=15)
    #sched2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 55], gamma=0.1)
    # sched2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - 5, eta_min=1e-6)
    #scheduler = optim.lr_scheduler.ChainedScheduler([sched1, sched2], optimizer=optimizer)
    train_loader, val_loader, sampler = prep_loaders(exp, season, rank, world_size, weekly=weekly, ddp=ddp)
    loader_len = len(train_loader.dataset) + (len(train_loader.dataset) % world_size) if ddp else len(train_loader.dataset)
    
    if rank == 0: helper = TrainHelper(model_name, tag=season + '_big', hyperparams=hps)
    if ddp: cpu_grp = dist.new_group(backend='gloo')
    if ddp: dist.barrier()
    stop_signal = torch.tensor([0], dtype=torch.int)

    try:
        t1 = time.time()
        for epoch in range(1, n_epochs + 1):
            #tx = time.time()
            if ddp: sampler.set_epoch(epoch)
            l = train(model, local_rank, train_loader, optimizer, loss_fn, epoch)
            if ddp:
                all_loss = [torch.zeros_like(l, device=torch.device('cpu')) for _ in range(world_size)]
                dist.all_gather(all_loss, l, group=cpu_grp)
                if rank == 0:
                    val_loss = validate(model, local_rank, val_loader, loss_fn, ddp=ddp)
                    train_loss = torch.sum(torch.tensor(all_loss)) / loader_len
            else:
                val_loss = validate(model, local_rank, val_loader, loss_fn, ddp=ddp)
                train_loss = l / loader_len
            scheduler.step()

            if ddp: dist.barrier()  # make sure all training in epoch is done before saving
            if rank == 0:  # this should work for dpp or not
                helper.add(train_loss, val_loss)
                helper.print(epoch)
                if ddp:
                    stop_signal = helper.checkpoint(model.module, epoch)
                else:
                    stop_signal = helper.checkpoint(model, epoch, search)

            if ddp: dist.barrier()  # make sure to not move forward with training while saving in rank0

            if ddp: dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM, group=cpu_grp)
            if stop_signal.item() > 0:
                # print(f'Stopping at epoch {epoch}', flush=True) # will this work or error out
                print(f'Reducing lr at epoch {epoch}', flush=True)
                # this has a builtin method but probs not work for DDP?
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
                helper.reset(epoch)

            #ty = time.time()
            #print(f'Epoch {epoch} done in {ty - tx} seconds', flush=True)

        # this doesnt work...
        # if rank > 0: exit(0)

        t2 = time.time()
        print(f'Run completed in {t2 - t1}', flush=True)

    except Exception as e:
        print(f'[ERROR]: Main on {local_rank} @ {time.time()} -- {e}')
    finally:
        if rank == 0: helper.finish(search)
        cleanup(ddp)

# =================================================================================
def train(model, device, train_loader, optimizer, loss_fn, epoch):
    try:
        train_loss = 0
        model.train()
        for i, (source, target, _) in enumerate(train_loader):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(source)
            loss = loss_fn(out, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        return torch.tensor(train_loss) 
    except Exception as e:
        print(f'[ERROR]: Training on {device} @ {time.time()} -- {e}', flush=True)
        raise e
        return

# ---------------------------------------------------------------------------------
def validate(model, device, val_loader, loss_fn, ddp=False):
    try:
        vloss = 0
        model.eval()
        val_model = model if not ddp else model.module
        with torch.no_grad():
            for i, (source, target, _) in enumerate(val_loader):
                source, target = source.to(device), target.to(device)
                out = val_model(source)
                loss = loss_fn(out, target)
                vloss += loss.item()

        vloss /= len(val_loader.dataset)
        return vloss
    except Exception as e:
        print(f'[ERROR]: Validating on {device} @ {time.time()} -- {e}', flush=True)
        raise e
        return

# ---------------------------------------------------------------------------------
def loss_fn(out, target):
    # original: MSE + penalty for negative predictions
    # return F.mse_loss(out, target, reduction='mean') + torch.mean(F.relu(-1 * out))

    # test 1: MSE + negative penalty + emph on discrepancies of non-zero parts.?
    mse_nonzero = F.mse_loss(out, target, weight=target)  # if target[i, j] is 0 -> loss[i, j] == 0, i.e. can't make gains from getting 0s right
    mse_all = F.mse_loss(out, target)  # this just mse everywhere
    pen_neg = torch.mean(F.relu(-1 * out))
    # print(f'Loss magnitudes: MSE Non0 {mse_nonzero.item()}, MSE All: {mse_all.item()}, Neg Penalty {pen_neg.item()}')
    # remove all except the nonzero?
    return mse_nonzero + mse_all + pen_neg

    # test 2: MSE + KL div? how would that work.?

# ---------------------------------------------------------------------------------
def prep_optimizer(model_params, lr=1e-4, wd=1e-2):
    # might we want different optimizer 
    # what about lr? with distributed training seen debates on scaling this
    optimizer = optim.AdamW(model_params, lr=lr, weight_decay=wd)  # use default betas?

    return optimizer

# ---------------------------------------------------------------------------------
def prep_loaders(exp, season, rank, world_size, weekly=False, ddp=False):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    
    train_ds = PrecipDataset('train', exp, season, weekly=weekly)
    val_ds = PrecipDataset('val', exp, season, weekly=weekly)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if ddp else None

    if sampler:
        train_loader = DataLoader(train_ds, batch_size=None, sampler=sampler, num_workers=n_workers, prefetch_factor=1)
    else:
        train_loader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=n_workers, prefetch_factor=1)

    val_loader = DataLoader(val_ds, batch_size=None, num_workers=n_workers, prefetch_factor=1)

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

