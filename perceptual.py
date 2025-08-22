import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def main():
    sn = SqueezeNet()
    return

class SqueezeNet(nn.Module):
    """SqueezeNet/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """
    def __init__(self, conv_index: str = 'mid'):
        super(SqueezeNet, self).__init__()
        weights = torchvision.models.SqueezeNet1_0_Weights.DEFAULT
        features = torchvision.models.squeezenet1_0(weights=weights).features
        self.transforms = weights.transforms()
        modules = [m for m in features]
        # 0: Conv, 1: ReLU, 2: MaxPool, 3-5: "Fire", 6: MaxPool, 7-10: "Fire", 11: MaxPool, 12: Fire
        if conv_index == 'start':
            self.sn = nn.Sequential(*modules[:3])
        elif conv_index == 'mid':
            self.sn = nn.Sequential(*modules[:8])
        elif conv_index == 'late':
            self.sn = nn.Sequential(*modules[:11])
        elif conv_index == 'full':
            self.sn = nn.Sequential(*modules)


        #vgg_mean = (0.485, 0.456, 0.406)
        #vgg_std = (0.229, 0.224, 0.225)
        #self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.sn.requires_grad = False

    def _fft(self, pred, target):
        p = pred - torch.mean(pred, dim=(2, 3), keepdim=True)
        t = target - torch.mean(target, dim=(2, 3), keepdim=True)
        fft_pred = torch.abs(torch.fft.fft2(p, norm='ortho'))
        fft_target = torch.abs(torch.fft.fft2(t, norm='ortho'))
        fft_wt = torch.log(1 + F.l1_loss(fft_pred, fft_target, reduction='none'))
        fft_loss = F.l1_loss(fft_pred, fft_target, weight=fft_wt)
        return fft_loss

    def _pcc(self, pred, target):
        mt = torch.mean(target, dim=(2, 3), keepdim=True)
        ts = target - mt
        mp = torch.mean(pred, dim=(2, 3), keepdim=True)
        ps = pred - mp
        eps = 0
        pcc_loss = torch.mean(torch.sum(ps * ts, dim=(2, 3)) / torch.sqrt(torch.sum(ps ** 2, dim=(2, 3)) * torch.sum(ts ** 2, dim=(2, 3)) + eps))
        return 1 - pcc_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SqueezeNet/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        def _forward(x):
            #x = self.sub_mean(x)
            x = self.sn(x)
            return x

        mse = F.mse_loss(pred, target)
        #fft = self._fft(pred, target)
        #mae = F.l1_loss(pred, target)
        #pcc = self._pcc(pred, target)

        _pred = torch.cat([pred] * 3, dim=1)
        _target = torch.cat([target] * 3, dim=1)
        #_pred = self.transforms(_pred)
        #_target = self.transforms(_target)
        _pred = _forward(_pred)  # should this also be with torch.no_grad() when validating.?

        with torch.no_grad():
            _target = _forward(_target.detach())

        perceptual = F.mse_loss(_pred, _target)

        return 1e-3 * perceptual + mse

if __name__ == '__main__':
    main()
