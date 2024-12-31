import torch

def main():
    pass

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

if __name__ == '__main__':
    main()

