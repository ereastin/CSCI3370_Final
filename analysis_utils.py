import numpy as np
import matplotlib.pyplot as plt
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeat

# ---------------------------------------------------------------------------------
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

# ---------------------------------------------------------------------------------
def pcc(a, b):
    a_mn = np.mean(a)
    b_mn = np.mean(b)
    return np.sum((a - a_mn) * (b - b_mn)) / np.sqrt(np.sum((a - a_mn) ** 2) * np.sum((b - b_mn) ** 2))

# ---------------------------------------------------------------------------------
def calc_daily_bias(a, b):
    # inputs are (B, 1, 96, 96)
    n_days = a.shape[0] // 4
    step_bias = a - b
    day_bias = torch.zeros((n_days,) + tuple(a.shape[1:]))
    for i in range(n_days):
        slc = slice(i * 4, (i + 1) * 4)
        day_bias[i] = torch.sum(step_bias[slc], dim=0)

    return torch.mean(day_bias, dim=0).squeeze(0)  # reduce dim 0 for monthly mean bias?

# ---------------------------------------------------------------------------------
def plot_bias(monthly_bias, extent, coords, model_name):
    """
    DJF winter, MAM spring, JJA summer, SON autumn
    test years are not shuffled
    """
    # these are currently the mean 6-hourly bias over each season..
    # want mean daily bias? mean monthly? mean seasonal?
    # Lin 2017 use mm per day bias, do that?
    monthly_bias = torch.stack(monthly_bias, dim=0)
    test = monthly_bias[:12] + monthly_bias[12:]
    print(f'Monthly average bias mm per day {torch.mean(test, dim=(1, 2))}')
    sel_sp = [2, 3, 4, 14, 15, 16]
    sel_su = [5, 6, 7, 17, 18, 19]
    sel_a = [8, 9, 10, 20, 21, 22]
    sel_w = [11, 0, 1, 23, 12, 13]
    spring = torch.mean(monthly_bias[sel_sp], dim=0).numpy(force=True)
    summer = torch.mean(monthly_bias[sel_su], dim=0).numpy(force=True)
    autumn = torch.mean(monthly_bias[sel_a], dim=0).numpy(force=True)
    winter = torch.mean(monthly_bias[sel_w], dim=0).numpy(force=True)

    datas = [spring, summer, autumn, winter]
    titles = ['MAM', 'JJA', 'SON', 'DJF']
    lvls = np.linspace(-2, 2, 17)
    lons, lats = coords[0], coords[1]
    cmap = 'BrBG'
    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(8, 8))
    axs = axs.flatten()
    fig.suptitle(f'Seasonal Precipitation Bias: {model_name}')
    for i, (ax, data, title) in enumerate(zip(axs, datas, titles)):
        cs = ax.contourf(lons, lats, data, lvls, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        ax.set_extent(extent, crs=ccrs.Geodetic())
        ax.add_feature(cfeat.STATES)
        ax.set_title(title)

    fig.colorbar(cs, ax=axs, orientation='horizontal', fraction=0.046, pad=0.04, label=r'Precip Bias [mm$\cdot$day$^{-1}$]')
    img_fname = f'biases.png'
    img_path = os.path.join(f'./models/{model_name}/', img_fname)
    plt.savefig(img_path, dpi=300, bbox_inches='tight')  # still too much space between rows
    print(f'Seasonal bias plotted to {img_path}')

# ---------------------------------------------------------------------------------
def plot_precip(sel_out, sel_y, sel_t, extent, coords, model_name):
    # colormap stuff
    clrs = ['#FFFFFF','#BEFFFF','#79C8FF','#3E62FF','#2F2DDE','#79DA62','#58D248','#3BBF3D','#28A83A','#F8FB64', '#FFD666','#FFA255','#FF6039','#F61F1F','#CD3B3B','#AC3333','#CD1599','#C725E0']
    cmap = colors.LinearSegmentedColormap.from_list('p_cmap', clrs, 18)
    cmap.set_bad(color='white')
    cmap.set_under(color='white')

    lvls = np.linspace(0, 32, 17)
    lons, lats = coords[0], coords[1]
    days = [17, 7, 18, 21]
    
    for i, (output, target, t_str) in enumerate(zip(sel_out, sel_y, sel_t)):
        low = (days[i] - 1) * 4
        high = low + 4
        sel_dates = slice(low, high)
        outs, targs = output[sel_dates].squeeze(1), target[sel_dates].squeeze(1)
        # so these are now (B, 1, 96, 96) -> (len(sel_dates), 96, 96)

        outs = list(torch.split(outs, 1, dim=0))
        targs = list(torch.split(targs, 1, dim=0))
        outs = [im.squeeze(0).numpy(force=True) for im in outs]
        targs = [im.squeeze(0).numpy(force=True) for im in targs]
        datas = outs + targs 
        rmses = [rmse(o, t) for (o, t) in zip(outs, targs)]
        pccs = [pcc(o, t) for (o, t) in zip(outs, targs)]

        fig, axs = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(13, 6))
        axs = axs.flatten()
        fig.suptitle(f'Predicted vs. Observed Data on {t_str}: {model_name}')
        for j, (ax, data) in enumerate(zip(axs, datas)):
            cs = ax.contourf(lons, lats, data, lvls, transform=ccrs.PlateCarree(), cmap=cmap, extend='max')
            ax.set_extent(extent, crs=ccrs.Geodetic())
            ax.add_feature(cfeat.STATES)
            if j < 4:
                ax.set_title(f'RMSE: {rmses[i]:.3f}, PCC: {pccs[i]:.3f}', fontsize=6)

        fig.colorbar(cs, ax=axs, orientation='horizontal', fraction=0.046, pad=0.04, label='Accumulated Precip [mm]')
        img_fname = f'{t_str[:4]}-{t_str[-2:]-{str(days[i]).zfill(2)}}.png'
        img_path = os.path.join(f'./models/{model_name}/', img_fname)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')  # still too much space between rows

    print('Precipitation figures plotted')

# ---------------------------------------------------------------------------------
def vis_attention(model, test_loader, device):
    # just take random input
    model = model.to(device)
    model.eval()
    upsmpl = nn.Upsample((96, 96), mode='trilinear')
    # this the right way to do this?
    def norm(t):
        t -= torch.min(t)
        t /= torch.max(t)
        return t

    for i, (x, y, t) in enumerate(test_loader):
        idx = torch.randint(x.shape[0] - 1, (1,))
        x = x.to(device)[idx]
        y = y.to(device)[idx].view(96, 96).numpy(force=True)
        out = model(x, test=True)
        attn_wts = model.attn_wts
        # attn_wts = [upsmpl(a) for a in attn_wts]
        [print(a) for a in attn_wts]
        return
        fig, axs = plt.subplots(3, 2)
        joint_attn = [attn_wts[0]]
        
        #for j in range(1, len(attn_wts)):
            #joint_attn.append(torch.matmul(attn_wts[j], joint_attn[j - 1]))
        #joint_attn = [norm(ja) for ja in joint_attn]
        #attn_wts = [norm(a) for a in attn_wts]
        for attn, ax in zip(attn_wts, axs.flat):
            #ax.imshow(attn.view(96, 96).numpy(force=True))
            ax.imshow(attn.view(attn.shape[2], attn.shape[3]).numpy(force=True))
        axs[-1][-2].imshow(y)
        axs[-1][-1].imshow(out.view(96, 96).numpy(force=True))
        plt.savefig('attn_wts.png', dpi=300, bbox_inches='tight')
        break

