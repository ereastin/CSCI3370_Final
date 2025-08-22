import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MERRA_VARS = ['QV', 'U', 'V', 'OMEGA', 'H', 'T']
LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])
## ================================================================================
def main():
    fig, ax = plt.subplots()
    season = 'spr'
    metric = 'ets'
    stats = []
    for v in MERRA_VARS:
        df = pd.read_csv(f'cus_{season}_{v}_shuffle.csv', index_col=0)
        for index, data in df.iterrows():
            stats.append({
                'label': index,
                'mean': data[f'{metric}_imp_mean'],
                'med': data[f'{metric}_imp_med'],
                'q1': data[f'{metric}_imp_25p'],
                'q3': data[f'{metric}_imp_75p'],
                'whislo': data[f'{metric}_imp_25p'],
                'whishi': data[f'{metric}_imp_75p']
            })
    stats = sorted(stats, key=lambda x: x['med'], reverse=True)
    stats = stats[:25]
    ax.bxp(stats, showfliers=False, showmeans=True, meanline=True)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha='right', rotation_mode='anchor')
    a = 'JJA' if season == 'sum' else 'MAM'
    ax.set(title=f'Feature Importance for {a}', ylabel=f'{metric.upper()} Importance', ylim=(-.05, .35))
    plt.axhline(y=0, color='gray', linewidth=1, linestyle='--')
    plt.savefig(f'./{season}_{metric}_bxplt.png', dpi=300)

## ================================================================================
def combine():
    v = 'V'
    df_add = pd.read_csv(f'./cus_sum_{v}_shuffle_add.csv', index_col=0)
    df_add.drop(['QV7', 'QV3', 'U7', 'U3'], inplace=True)
    df = pd.read_csv(f'./cus_sum_{v}_shuffle.csv', index_col=0)
    out = pd.concat([df, df_add])
    print(out)
    out.to_csv(f'./cus_sum_{v}_shuffle_all.csv')
    return

def log_to_csv():
    v = 'QV'
    with open(f'./_logs/{v}shuf_sum_738756.out', 'r') as f:
        content = f.read()
        chunks = content.split('*' * 60)[1:-1]

    df_list = []
    for c in chunks:
        print(c)
        lines = c.split('\n')
        a = eval(lines[1])
        df_idx = a[0]['var'] + str(a[0]['levels'])
        mean_mse = float(lines[2].split(':')[-1].strip())
        mean_pcc = float(lines[3].split(':')[-1].strip())
        n_good_pccs = int(lines[4].split(':')[-1].split('of')[0].strip())
        mean_ets = float(lines[5].split(':')[-1].strip())
        bias = float(lines[6].split(':')[-1].strip())
        #rmse_met = lines[8].split()
        [rmn, rmd, r25, r75] = [float(s.strip()) for s in lines[8].split()]
        #ets_met = lines[9].split()
        [emn, emd, e25, e75] = [float(s.strip()) for s in lines[9].split()]
        df = pd.DataFrame({
            'mean_mse': mean_mse,
            'mean_pcc': mean_pcc,
            'n_good_pcc': n_good_pccs,
            'mean_ets': mean_ets,
            'rmse_imp_mean': rmn,
            'rmse_imp_med': rmd,
            'rmse_imp_25p': r25,
            'rmse_imp_75p': r75,
            'ets_imp_mean': emn,
            'ets_imp_med': emd,
            'ets_imp_25p': e25,
            'ets_imp_75p': e75,
            'mean_reg_bias': bias
        }, index=[df_idx])
        df_list.append(df)

    out = pd.concat(df_list)
    print(out)
    out.to_csv(f'./cus_sum_{v}_shuffle.csv')

## ================================================================================
if __name__ == '__main__':
    main()
