import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

def main():
    th = TrainHelper('a')

# ---------------------------------------------------------------------------------
class TrainHelper:
    def __init__(self, model_name, tag='', hyperparams={}, save_every=5):
        self.tag = tag
        self.hyperparams = hyperparams
        self.save_every = save_every
        self.model_path = f'/home/eastinev/AI/incept/models/{model_name}/'
        self.train_loss = []
        self.val_loss = []
        self._improve_thresh = 100
        self._base_epoch = 0
        self._last_save = 0
        self._save_hyperparams()
    
    def add(self, train, val):
        if isinstance(train, torch.Tensor): train = train.item()
        if isinstance(val, torch.Tensor): val = val.item()
        self.train_loss.append(train)
        self.val_loss.append(val)

    def _best_loss(self):
        print(f'Hyperparameters: \n{self.hyperparams}')
        print(f'Min train loss: {np.amin(self.train_loss)} @ epoch {np.argmin(self.train_loss) + 1}')
        print(f'Min val loss: {np.amin(self.val_loss)} @ epoch {np.argmin(self.val_loss) + 1}')
        print('*' * 60)

    def _plot_loss(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(len(self.train_loss)), self.train_loss, label='train loss')
        ax.plot(np.arange(len(self.val_loss)), self.val_loss, label='val loss')
        ax.set(xlabel='epoch', ylabel='loss', ylim=(0, 2 * np.amax(self.train_loss)))
        plt.legend()
        fig.tight_layout()
        sv_pth = os.path.join(self.model_path, f'loss_{self.tag}.png')
        plt.savefig(sv_pth, dpi=300.0)

    def _save_hyperparams(self):
        sv_pth = os.path.join(self.model_path, f'hyperparams_{self.tag}.json')
        with open(sv_pth, 'w') as f:
            json.dump(self.hyperparams, f, indent=4)

    def finish(self, search):
        try:
            if search:
                self._best_loss()
            else:
                self._plot_loss()
        except ValueError as e:
            print('Zero-length array, can\'t perform requested operations')

    def checkpoint(self, model, epoch, search=False):
        if self._stop_early(epoch):
            return torch.tensor([1], dtype=torch.int)
        if self._new_min and not search:
            self._save_point(model, epoch)
        return torch.tensor([0], dtype=torch.int)

    def _save_point(self, model, epoch):
        state = model.state_dict()
        sv_pth = os.path.join(self.model_path, f'params_{self.tag}.pth')
        torch.save(state, sv_pth)
        self._last_save = epoch
        print(f'Model state saved at epoch {epoch}')

    def _stop_early(self, epoch):
        if self._improve_thresh == 0: return False
        if epoch < (self._improve_thresh + self._base_epoch): return False
        if epoch < (self._improve_thresh + self._last_save): return False
        smooth = np.convolve(self.val_loss[-(self._improve_thresh):], np.array([1/10, 1/5, 2/5, 1/5, 1/10]), 'valid')
        check = np.mean(np.gradient(smooth))
        return True if check >= 0.0 else False

    def reset(self, epoch):
        self._base_epoch = epoch

    @property
    def _new_min(self):
        if np.amin(self.val_loss) == self.val_loss[-1]:
            return True
        else:
            return False

    def print(self, epoch):
        t_loss = self.train_loss[-1]
        v_loss = self.val_loss[-1]
        print(f'Epoch {epoch} | Train loss: {t_loss}, Val loss: {v_loss}')

if __name__ == '__main__':
    main()

