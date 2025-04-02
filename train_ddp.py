import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# import init_process_group, destroy_process_group, all_gather, all_reduce, ReduceOp.SUM, is_initialized, new_group, barrier
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
from PrecipDataset import PrecipDataset

# =================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('mode', type=str)
    parser.add_argument('n_epochs', type=int)
    parser.add_argument('-r', '--rmvar', type=str)
    parser.add_argument('-z', '--zero', action='store_true')
    parser.add_argument('-p', '--precip', action='store_true')
    parser.add_argument('-b', '--bias', action='store_true')
    parser.add_argument('-s', '--search', action='store_true')
    parser.add_argument('-t', '--timeit', action='store_true')
    parser.add_argument('-d', '--ddp', action='store_true')
    parser.add_argument('-op', '--optimparams', type=str)
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    args = parser.parse_args()

    model_name, mode, n_epochs, ddp = args.model_name, args.mode, args.n_epochs, args.ddp
    rm_var, zero = args.rmvar, args.zero
    exp, season = args.exp, args.season
    op = args.optimparams.split(':')
    lr, wd = float(op[0]), float(op[1])
    # TODO: what about removing vertical shear? set all to..? average of midlevel?
    note = 'ctrl' if rm_var is None else f'{"no" if zero else "mn"}{rm_var}'
    weekly = False

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
        rank = 0  # this so we can use the conditionals below
        local_rank = torch.device('cuda')

    model = IRNv4_3DUNet(6, depth=35, Na=1, Nb=2, Nc=1, drop_p=0.3).to(local_rank)
    if ddp: model = DDP(model, device_ids=[local_rank])

    optimizer = prep_optimizer(model, lr, wd)
    sched1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - 10, eta_min=1e-5)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[10])
    train_loader, val_loader, sampler = prep_loaders(exp, season, rank, world_size, weekly=weekly, ddp=ddp)
    loader_len = len(train_loader.dataset) + (len(train_loader.dataset) % world_size) if ddp else len(train_loader.dataset)
    
    if rank == 0: helper = TrainHelper(model_name, tag=season)
    if ddp: cpu_grp = dist.new_group(backend='gloo')
    if ddp: dist.barrier()
    stop_signal = torch.tensor([0], dtype=torch.int) 

    try:
        t1 = time.time()
        for epoch in range(1, n_epochs + 1):
            if ddp: sampler.set_epoch(epoch)
            l = train(model, local_rank, train_loader, optimizer, epoch)
            if ddp:
                all_loss = [torch.zeros_like(l, device=torch.device('cpu')) for _ in range(world_size)]
                dist.all_gather(all_loss, l, group=cpu_grp)
                if rank == 0:
                    # TODO: worth splitting this guy across multiple GPUs too? need drop_last=True so no repeats 
                    # barrier here for validation?
                    val_loss = validate(model, local_rank, val_loader)
                    train_loss = torch.sum(torch.tensor(all_loss)) / loader_len
            else:
                val_loss = validate(model, local_rank, val_loader)
                train_loss = l / loader_len
            scheduler.step()

            if ddp: dist.barrier()  # make sure all training in epoch is done before saving
            if rank == 0:  # this should work for dpp or not
                helper.add(train_loss, val_loss)
                helper.print(epoch)
                stop_signal = helper.checkpoint(model.module, epoch)  # this.. will not work if not ddp yes? same w/validate()

            if ddp: dist.barrier()  # make sure to not move forward with training while saving in rank0

            dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM, group=cpu_grp)
            if stop_signal.item() > 0:
                print(f'Stopping at epoch {epoch}', flush=True) # will this work or error out
                break

        # this work right?
        if rank > 0: exit(0)

        t2 = time.time()
        print(f'Run completed in {t2 - t1}', flush=True)

    except Exception as e:
        print(f'[ERROR]: {local_rank} @ {time.time()} -- {e}')
    finally:
        if rank == 0: helper.plot_loss()
        cleanup(ddp)

# =================================================================================
def train(model, device, train_loader, optimizer, epoch):
    try:
        train_loss = 0
        model.train()
        # print(device, 'train start', time.time(), flush=True)
        for i, (source, target, _) in enumerate(train_loader):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(source)
            loss = F.mse_loss(out, target, reduction='mean') + torch.mean(F.relu(-1 * out))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # print(device, 'train end', time.time(), flush=True)
        return torch.tensor(train_loss) 
    except Exception as e:
        print(f'[ERROR]: {device} @ {time.time()} -- {e}', flush=True)
        return

# ---------------------------------------------------------------------------------
def validate(model, device, val_loader):
    try:
        # print(device, 'val start', time.time(), flush=True)
        vloss = 0
        model.eval()
        with torch.no_grad():
            for i, (source, target, _) in enumerate(val_loader):
                source, target = source.to(device), target.to(device)
                out = model.module(source)
                loss = F.mse_loss(out, target, reduction='mean') + torch.mean(F.relu(-1 * out))
                vloss += loss.item()

        # print(device, 'val end', time.time(), flush=True)
        vloss /= len(val_loader.dataset)
        return vloss
    except Exception as e:
        print(f'[ERROR]: {device} @ {time.time()} -- {e}', flush=True)
        return

# ---------------------------------------------------------------------------------
def prep_optimizer(model, lr=1e-4, wd=1e-2):
    # might we want different optimizer 
    # what about lr? with distributed training seen debates on scaling this
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)  # use default betas?

    return optimizer

# ---------------------------------------------------------------------------------
def prep_loaders(exp, season, rank, world_size, weekly=False, ddp=False):
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    
    train_ds = PrecipDataset('train', exp, season, weekly=weekly)
    val_ds = PrecipDataset('val', exp, season, weekly=weekly)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if ddp else None
    
    train_loader = DataLoader(train_ds, batch_size=None, sampler=sampler, num_workers=n_workers)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=n_workers)

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

