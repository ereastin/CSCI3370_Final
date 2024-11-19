import torch
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from PrecipDataset import PrecipDataset
from Networks import *

def main():
    if len(sys.argv) < 3:
        print('Missing args')
        sys.exit(20)

    model_name, mode = sys.argv[1], sys.argv[2]  # cnn, unet, aunet, cae; train, test, val

    if torch.cuda.is_available():
        print('Using GPU')
        device = torch.device('cuda')
    else:
        print('No GPU available, exiting..')
        device = torch.device('cpu')
        #sys.exit(21)

    print('device secured')
    # disable automatic batch, doing this in PrecipDataset
    loader = DataLoader(PrecipDataset(mode), batch_size=None, pin_memory=True)
    print('loader secured')
    in_channels = 64  # 8 variables at 8 pressure heights
    est_batch_size = 120  # 4xdaily for 30 days
    h, w = 96, 96  # 96x96 grid
    if model_name == 'cnn':
        model = CNN(in_channels)
    elif model_name == 'unet':
        model = UNet(in_channels)
    elif model_name == 'aunet':
        model = AttentionUNet(in_channels)
    elif model_name == 'cae':
        model = CAE_LSTM(in_channels)
    else:
        print(f'Model "{model_name}" not valid')

    # run multiple GPUs if possible
    # pytorch says something about using distributed version.. not sure how take a look
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)

    # summary(model, input_size=(est_batch_size, in_channels, h, w))
    path = os.path.join(f'./models/{model_name}_params.pth')

    if mode == 'train':
        # adjust these settings
        t1 = time.time()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=[0.9, 0.99])
        model_loss = train_model(model, optimizer, loader, device, epochs=1)
        print(f'best loss: {min(model_loss)}')
        t2 = time.time()
        print(f'Time for 1 epoch: {t2 - t1}')
        plot_loss(model_loss)
    
        save_model_params(model, path)
        return
    
    elif mode in ['test', 'val']:
        model = load_model(model, path)
        print(f'Checking accuracy on {mode} dataset')
        check_accuracy(model, model_name, loader, device)
        return
    else:
        print(f'No mode "{mode}"')
        return

def save_model_params(model, path):
    torch.save(model.state_dict(), path)
    return

def load_model(model, path):
    model.load_state_dict(torch.load(path), weights_only=False)
    model.eval()
    return model

def plot_loss(loss_list):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss_list)), loss_list, label='loss')
    ax.set(xlabel='iteration', ylabel='loss')
    plt.legend()
    plt.show()  # will this work with x11

def check_accuracy(model, model_name, loader, device):
    model = model.to(device)
    with torch.no_grad():
        for x, y, t_str in loader:
            x = x.to(device=device)  # move to device, e.g. GPU -> may need to specify dtype here..?
            y = y.to(device=device)
            output = model(x)
            # recall these are values for one month of data!
            mse = F.mse_loss(output, y)  # this should be one value as mean mse for all days in the month..
            rmse = torch.sqrt(mse)
            sel_dates = [0, -1]  # pick first/last days of month, add more?
            out_check, y_check = output[sel_dates], y[sel_dates]
            fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
            fig.suptitle('Predicted vs. Observed Data: {model_name}')
            for i in range(len(sel_dates)):
                ax[i][0].imshow(out_check[i])
                ax[i][1].imshow(y_check[i])

            ax[0][0].set(title='Predicted Precip')
            ax[0][1].set(title='Observed Precip')
            ax[-1][0].set(xlabel='Longitude, deg W', ylabel='Latitude, deg N')
            fig.tight_layout()
            img_fname = f'{t_str}_sel.png'
            img_path = os.path.join('./models/{model_name}_outputs/')
            plt.savefig()
    return


def train_model(model, optimizer, loader, device, loss_func=F.mse_loss, epochs=1):
    """
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: loss_list
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    loss_list = []
    print_every = 1  # loader is len 16*12 not huge
    for e in range(epochs):
        for t, (x, y, t_str) in enumerate(loader):
            if x.shape[0] != y.shape[0]:
                print(f'Incorrect dimensions for dates {t_str}: source {x.shape} vs. target {y.shape}')
                continue

            t1 = time.time()
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
            t2 = time.time()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                print(f'Dates {t_str} completed in: {t2 - t1}')
                print()

    return loss_list

if __name__ == '__main__':
    main()

