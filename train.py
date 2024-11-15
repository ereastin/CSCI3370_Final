import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from PrecipDataset import PrecipDataset
from Networks import *

def main():
    mode = sys.argv[1]  # train, test, val
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No GPU available, exiting..')
        sys.exit(21)

    # disable automatic batch, doing this in PrecipDataset
    loader = DataLoader(PrecipDataset(mode), batch_size=None)

    in_channels = 64
    model = CNN(in_channels)
    #model = UNet(in_channels)
    #model = AttentionUNet(in_channels)
    #model = CAE_LSTM(in_channels)
    path = os.path.join('/scratch/eastinev/CNN_params.pth')

    if mode == 'train':
        # adjust these settings
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=[0.9, 0.99])
        model_loss = train_model(model, optimizer, loader, device, epochs=1)
        print(f'best loss: {min(model_loss)}')
        plot_loss(model_loss)

        x = input('Save model params?: ')
        if x == 'y':
            save_model_params(model)
        return
    
    elif mode in ['test', 'val']:
        model = load_model(model, path)
        print(f'Checking accuracy on {mode} dataset')
        check_accuracy(model, loader, device)
        return
    else:
        print(f'No mode "{mode}"')
        return

def save_model_params(model, path):
    torch.save(model.state_dict(), path)
    return

def load_model(model, path):
    model.load_state_dict(torch.load(path), weights_only=True)
    model.eval()
    return model

def plot_loss(loss_list):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss_list)), loss_list, label='loss')
    ax.set(xlabel='iteration', ylabel='loss')
    plt.legend()
    plt.show()  # will this work with x11

def check_accuracy(model, loader, device):
    # TODO: adjust this to pull out specific timesteps?
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)  # move to device, e.g. GPU -> may need to specify dtype here..?
            y = y.to(device=device)
            out = model(x)
            #err = out - y  # want to keep these associated with datetime?
            #std = np.std(err)
            #mn = np.mean(err)
            # store these/plot/etc.
    return


def train_model(model, optimizer, loader, device, loss_func=F.mse_loss, epochs=1):
    """
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: loss_list
    """
    # TODO: add some sort of timing so we can get a gauge on expected run time
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    loss_list = []
    print_every = 1  # loader is len 16*12 not huge
    for e in range(epochs):
        for t, (x, y) in enumerate(loader):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU -> may need to specify dtype here..?
            y = y.to(device=device)

            # check that this is right way to do things
            # would prefer loss out here but not sure if thats best
            # Zhang et al do loss/predict out here
            # probably prefer that?
            output = model(x)
            loss = loss_func(output, y)
            loss_list.append(loss.item())

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                print()

    return loss_list

if __name__ =='__main__':
    main()

