import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#import dask
#from dask import array
#dask.config.set(scheduler='threads', num_workers=6)
#dask.config.set({'temporary_directory': '/scratch/eastinev/tmp'})
# dask.config.set({"distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 1024})
#from dask.distributed import Client, LocalCluster
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
from Dummy import Dummy
from PrecipDataset import PrecipDataset
from OTPrecipDataset import OTPrecipDataset

# for Lazy module dry-runs.. handle this better for other input shapes
C, D, H, W = 6, 35, 80, 144
FROM_LOAD = False

# =================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('n_epochs', type=int)
    parser.add_argument('-s', '--search', action='store_true')
    parser.add_argument('-d', '--ddp', action='store_true')
    parser.add_argument('-op', '--optimparams', type=str)
    parser.add_argument('-e', '--exp', type=str)
    parser.add_argument('-sn', '--season', type=str)
    parser.add_argument('-t', '--tag', type=str, default='')
    args = parser.parse_args()

    model_name, n_epochs, ddp = args.model_name, args.n_epochs, args.ddp
    exp, season = args.exp, args.season
    tag = args.tag
    search = args.search
    if args.optimparams is not None:
        op = args.optimparams.split(':')
        lr, wd = float(op[0]), float(op[1])
    model_id_tag = season + tag
    weekly = True
    print(f'Job ID: {os.environ["SLURM_JOBID"]}', flush=True)

    if ddp:
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
        rank = 0
        local_rank = torch.device('cuda')

    if search:  # random search instead of grid search
        base = np.random.choice([8, 12, 16, 24, 32])
        lin_act = np.random.choice(np.append(np.linspace(0.01, 0.3, 20), np.array([1.0])))
        #Na = np.random.choice(np.arange(1, 6))
        #Nb, Nc = 2 * Na, Na
        # just force to 1 each
        Na, Nb, Nc = 1, 1, 1
        lr = np.random.choice(np.logspace(-5, -1, 20))
        wd = np.random.choice(np.append(np.logspace(-3, -1, 10), np.linspace(0, 0.5, 5)))
        drop_p = np.random.choice(np.linspace(0.1, 0.3, 10))
        bias = np.random.choice(np.array([True, False]))
        opt_type = np.random.choice(np.array(['adamw', 'adam', 'sgd']))
        hps = {
            'base': base, 'lin_act': lin_act, 'Na': Na, 'Nb': Nb, 'Nc': Nc,
            'optim': opt_type, 'lr': lr, 'wd': wd, 'drop_p': drop_p, 'bias': bias
        }
    else:
        base = 6
        lin_act = 0.3
        Na, Nb, Nc = 1, 1, 1
        lr = 1e-4
        #lr *= 4 if ddp else 1
        max_lr = 5e-3
        #max_lr *= 4 if ddp else 1
        wd = 0
        drop_p = 0.1
        bias = False
        opt_type = 'adamw'
        hps = {
            'base': base, 'lin_act': lin_act, 'Na': Na, 'Nb': Nb, 'Nc': Nc,
            'optim': opt_type, 'lr': lr, 'max_lr': max_lr, 'wd': wd, 'drop_p': drop_p, 'bias': bias
        }

    model = IRNv4_3DUNet(6, depth=35, Na=Na, Nb=Nb, Nc=Nc, base=base, bias=bias, drop_p=drop_p, lin_act=lin_act).to(local_rank).float()
    # model = Dummy().to(local_rank).float()

    if FROM_LOAD:
        model = load_model(model, model_name, model_id_tag, local_rank)
    else:
        # dry-run for Lazy modules -- must be called before DDP init
        model(torch.ones(1, C, D, H, W).to(local_rank))

    if ddp: model = DDP(model, device_ids=[local_rank])

    optimizer = prep_optimizer(model.parameters(), lr, wd, opt_type)
    #start_factor = 1 if search else 0.1
    #scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=5)
    #sched2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 55], gamma=0.1)
    # sched2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - 5, eta_min=1e-6)
    #scheduler = optim.lr_scheduler.ChainedScheduler([sched1, sched2], optimizer=optimizer)

    train_loader, val_loader, sampler = prep_loaders(exp, season, rank, world_size, weekly=weekly, ddp=ddp)
    loader_len = len(train_loader.dataset) + (len(train_loader.dataset) % world_size) if ddp else len(train_loader.dataset)

    onecycle = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=(n_epochs * loader_len))
    
    if rank == 0: helper = TrainHelper(model_name, tag=model_id_tag, hyperparams=hps)
    if ddp: cpu_grp = dist.new_group(backend='gloo')
    if ddp: dist.barrier()
    stop_signal = torch.tensor([0], dtype=torch.int)
    #print('pre loop', flush=True)

    try:
        '''
        nn_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        n_cpus -= 4
        print(n_cpus, nn_cpus)
        cluster = LocalCluster(n_workers=n_cpus, memory_limit='2GiB')  # have to specify this think it defaults to 4 or smthn
        print(cluster, flush=True)
        with Client(cluster) as client:
            print(client, flush=True)
            print(client.run(os.getenv, "MALLOC_TRIM_THRESHOLD_"))
        '''
        #_ = next(iter(train_loader))  # ig check this is ready?
        t1 = time.time()
        for epoch in range(1, n_epochs + 1):
            tx = time.time()
            if ddp: sampler.set_epoch(epoch)
            l = train(model, local_rank, train_loader, optimizer, loss_fn, scheduler=onecycle)
            if ddp:
                all_loss = [torch.zeros_like(l, device=torch.device('cpu')) for _ in range(world_size)]
                dist.all_gather(all_loss, l, group=cpu_grp)
                if rank == 0:
                    val_loss = validate(model, local_rank, val_loader, loss_fn, ddp=ddp)
                    train_loss = torch.sum(torch.tensor(all_loss)) / loader_len
            else:
                val_loss = validate(model, local_rank, val_loader, loss_fn, ddp=ddp)
                train_loss = l / loader_len
            # scheduler.step()

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
                print(f'No improvement, stopping at epoch {epoch}', flush=True) # will this work or error out
                break

            ty = time.time()
            print(f'Epoch {epoch} done in {ty - tx} seconds', flush=True)

        t2 = time.time()
        print(f'Run completed in {t2 - t1}', flush=True)

    except Exception as e:
        print(f'[ERROR]: Main on {local_rank} @ {time.time()} -- {e}')
        raise e
    finally:
        if rank == 0: helper.finish(search)
        cleanup(ddp)

# =================================================================================
def train(model, device, train_loader, optimizer, loss_fn, scheduler=None):
    try:
        train_loss = 0
        model.train()
        for i, (source, target, tt) in enumerate(train_loader):
            if source is None or target is None:
                continue
            source, target = source.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            out = model(source)
            loss = loss_fn(out, target)
            # if torch.isnan(torch.tensor(loss.item())):
            #     print(f'train {tt} loss is nan, {torch.isnan(source).any()}', loss.item(), flush=True)
            #     raise KeyboardInterrupt
            train_loss += loss.item()
            # print(f'train {tt}, {torch.isnan(source).any()}', loss.item(), flush=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
        return torch.tensor(train_loss)
    except Exception as e:
        print(f'[ERROR]: Training on {device} @ {time.time()} -- {e}', flush=True)
        raise e
        return

# ---------------------------------------------------------------------------------
def validate(model, device, val_loader, loss_fn, ddp=False):
    try:
        vloss = 0
        # ctr = 0
        model.eval()
        val_model = model if not ddp else model.module
        with torch.no_grad():
            for i, (source, target, _) in enumerate(val_loader):
                if source is None or target is None:
                    # ctr += 1
                    continue
                source, target = source.to(device).float(), target.to(device).float()
                out = val_model(source)
                loss = loss_fn(out, target)
                # print('val', loss.item(), flush=True)
                vloss += loss.item()

        vloss /= len(val_loader.dataset)
        return vloss
    except Exception as e:
        print(f'[ERROR]: Validating on {device} @ {time.time()} -- {e}', flush=True)
        raise e
        return

# ---------------------------------------------------------------------------------
def loss_fn(pred, target):
    # NOTE: potentially interesting thing here being how model performs with different loss fn
    # over/underpredicting, drizzle etc.
    # original: MSE + penalty for negative predictions
    # return F.mse_loss(out, target, reduction='mean') + torch.mean(F.relu(-1 * out))
    mae_loss = F.l1_loss(pred, target)
    return mae_loss

    # test 1: MSE + negative penalty + emph on discrepancies of non-zero parts.?
    # if target[i, j] is 0 -> loss[i, j] == 0, i.e. can't make gains from getting 0s right
    #mse_nonzero = F.mse_loss(out, target, weight=target)
    mse_all = F.mse_loss(pred, target)  # this just mse everywhere
    #pen_neg = torch.mean(F.relu(-1 * out))

    # TODO: what about a 2D FFT.? then could at least compare spectral content.?
    B = pred.shape[0]
    hpf = torch.zeros((B, 1, 3, 3))
    hpf[:, 0] = torch.tensor([[[0, -1/4, 0], [-1/4, 2, -1/4], [0, -1/4, 0]]])
    hpf = hpf.to('cuda')
    phf = F.conv2d(pred, hpf, padding='same')
    thf = F.conv2d(target, hpf, padding='same')
    hpf_mse = F.mse_loss(phf, thf)
    # print(hpf_mse.item())
    return hpf_mse + mse_all

    # would be interesting to rewrite into high/low frequency nets
    # e.g. pass LSF through high/low-pass filters then do sep nets and combine

    # test 3: PCC uncentered (no mean sub)... this require gaussian distr? centered def not good
    # perhaps can try centered in comb with mse_nonzero thing?
    d = (2, 3)
    # preserve dims for subtraction --> (B, 1, H, W)
    # out_diff = out - torch.mean(out, dim=d, keepdim=True)
    # target_diff = target - torch.mean(target, dim=d, keepdim=True)
    #pcc = torch.sum(out * target, dim=d) / torch.sqrt(torch.sum(out ** 2, dim=d) * torch.sum(target ** 2, dim=d))
    return mse_all + mse_nonzero  #1 - torch.mean(pcc) + mse_all #+ mse_nonzero  # pcc near 1 is good if pcc is negative then this is even larger

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
    prefetch = 2
    train_ds = OTPrecipDataset('train', exp, season, weekly=weekly)
    val_ds = OTPrecipDataset('val', exp, season, weekly=weekly)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if ddp else None

    if sampler:
        train_loader = DataLoader(train_ds, batch_size=None, sampler=sampler, num_workers=n_workers, prefetch_factor=prefetch)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=None,
            shuffle=True,
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

