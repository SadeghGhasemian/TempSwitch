"""
TOJ RT plots (RSI × history, SOA × history, Block × history)
Adjusted x‑axis margin for SOA plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------- Configuration --------------------
base_dir = Path('E:/TemporalSwitch/Pymc_reaction_time')
data_file = base_dir / 'toj_rt.csv'
output_dir = base_dir / 'output_toj_raw'
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.2)

# -------------------- Load and prepare data --------------------
data = pd.read_csv(data_file)
data.columns = data.columns.str.lower()

print("Column names:", data.columns.tolist())
print("First 5 rows:\n", data.head())

# Convert rsi to ms if needed (assume seconds if max < 100)
if data['rsi'].max() < 100:
    print("Converting rsi from seconds to ms")
    data['rsi'] = data['rsi'] * 1000
# SOA may already be ms; adjust if values are very small (e.g., < 1)
if 'soa' in data and data['soa'].max() < 100:
    print("Converting soa from seconds to ms")
    data['soa'] = data['soa'] * 1000

# History levels
hist_vals = data['hist'].unique()
print("Unique history values:", hist_vals)
if len(hist_vals) != 2:
    raise ValueError(f"Expected 2 history levels, got {len(hist_vals)}: {hist_vals}")

data['hist'] = pd.Categorical(data['hist'], categories=hist_vals, ordered=True)
hist_map = {hist_vals[0]: 'repeat', hist_vals[1]: 'switch'}
data['hist'] = data['hist'].cat.rename_categories(hist_map)
print("Mapped history to:", data['hist'].cat.categories.tolist())

subjects = data['subject'].unique()
print(f"Number of subjects: {len(subjects)}")

rsi_levels = np.sort(data['rsi'].unique())
soa_levels = np.sort(data['soa'].unique()) if 'soa' in data else np.array([])
block_levels = np.sort(data['block'].unique()) if 'block' in data else np.array([])

print("RSI levels:", rsi_levels)
if 'soa' in data:
    print("SOA levels:", soa_levels)
if 'block' in data:
    print("Block levels:", block_levels)

hist_levels = data['hist'].cat.categories

# -------------------- Helper function --------------------
def compute_subject_condition_means(df, condition_col, condition_levels):
    n_subj = len(subjects)
    n_cond = len(condition_levels)
    n_hist = len(hist_levels)
    rt_array = np.full((n_subj, n_cond, n_hist), np.nan)

    for i, subj in enumerate(subjects):
        for j, cond_val in enumerate(condition_levels):
            for k, hist_val in enumerate(hist_levels):
                mask = (df['subject'] == subj) & (df[condition_col] == cond_val) & (df['hist'] == hist_val)
                subset = df.loc[mask, 'rt']
                if not subset.empty:
                    rt_array[i, j, k] = subset.mean() * 1000   # convert rt to ms
    return rt_array

# -------------------- Plotting function --------------------
def plot_rt_vs_condition(condition_values, rt_repeat, rt_switch,
                          sem_repeat, sem_switch,
                          xlabel, ylim, xtick_rotation=0,
                          xmargin=50,
                          filename='plot.pdf'):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.errorbar(condition_values, rt_repeat, yerr=sem_repeat,
                fmt='-o', color='k', ecolor='k', elinewidth=1.2,
                capsize=0, markersize=9, markerfacecolor='k',
                linewidth=1.5, label='Repeat')
    ax.errorbar(condition_values, rt_switch, yerr=sem_switch,
                fmt='-o', color='.5', ecolor='.5', elinewidth=1.2,
                capsize=0, markersize=9, markerfacecolor='.5',
                linewidth=1.5, label='Switch')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Reaction time (ms)')
    ax.set_xticks(condition_values)
    ax.set_xticklabels([str(v) for v in condition_values], rotation=xtick_rotation)
    ax.set_ylim(ylim)
    ax.set_xlim(condition_values[0] - xmargin, condition_values[-1] + xmargin)
    ax.yaxis.set_major_locator(plt.MultipleLocator(100))
    ax.legend(loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# -------------------- RSI plot --------------------
if len(rsi_levels) > 0:
    rt_array_rsi = compute_subject_condition_means(data, 'rsi', rsi_levels)
    print("rt_array_rsi shape:", rt_array_rsi.shape)
    print("Non-NaN count:", np.sum(~np.isnan(rt_array_rsi)))

    rt_rep_rsi = rt_array_rsi[:, :, 0]
    rt_swi_rsi = rt_array_rsi[:, :, 1]

    mean_rep_rsi = np.nanmean(rt_rep_rsi, axis=0)
    mean_swi_rsi = np.nanmean(rt_swi_rsi, axis=0)
    n_rep = np.sum(~np.isnan(rt_rep_rsi), axis=0)
    n_swi = np.sum(~np.isnan(rt_swi_rsi), axis=0)
    sem_rep_rsi = np.nanstd(rt_rep_rsi, axis=0, ddof=0) / np.sqrt(n_rep)
    sem_swi_rsi = np.nanstd(rt_swi_rsi, axis=0, ddof=0) / np.sqrt(n_swi)

    plot_rt_vs_condition(rsi_levels, mean_rep_rsi, mean_swi_rsi,
                         sem_rep_rsi, sem_swi_rsi,
                         xlabel='RSI (ms)', ylim=(500,900), xtick_rotation=45,
                         xmargin=50, filename='TOJ_RSI_history.pdf')
else:
    print("No RSI data found.")

# -------------------- SOA plot --------------------
if 'soa' in data and len(soa_levels) > 0:
    rt_array_soa = compute_subject_condition_means(data, 'soa', soa_levels)
    print("rt_array_soa shape:", rt_array_soa.shape)
    print("Non-NaN count:", np.sum(~np.isnan(rt_array_soa)))

    rt_rep_soa = rt_array_soa[:, :, 0]
    rt_swi_soa = rt_array_soa[:, :, 1]

    mean_rep_soa = np.nanmean(rt_rep_soa, axis=0)
    mean_swi_soa = np.nanmean(rt_swi_soa, axis=0)
    n_rep = np.sum(~np.isnan(rt_rep_soa), axis=0)
    n_swi = np.sum(~np.isnan(rt_swi_soa), axis=0)
    sem_rep_soa = np.nanstd(rt_rep_soa, axis=0, ddof=0) / np.sqrt(n_rep)
    sem_swi_soa = np.nanstd(rt_swi_soa, axis=0, ddof=0) / np.sqrt(n_swi)

    plot_rt_vs_condition(soa_levels, mean_rep_soa, mean_swi_soa,
                         sem_rep_soa, sem_swi_soa,
                         xlabel='SOA (ms)', ylim=(500,1000),
                         xmargin=15, filename='TOJ_SOA_history.pdf')
else:
    print("No SOA column or empty levels.")

# -------------------- Block plot --------------------
if 'block' in data and len(block_levels) > 0:
    rt_array_blk = compute_subject_condition_means(data, 'block', block_levels)
    print("rt_array_blk shape:", rt_array_blk.shape)
    print("Non-NaN count:", np.sum(~np.isnan(rt_array_blk)))

    rt_rep_blk = rt_array_blk[:, :, 0]
    rt_swi_blk = rt_array_blk[:, :, 1]

    mean_rep_blk = np.nanmean(rt_rep_blk, axis=0)
    mean_swi_blk = np.nanmean(rt_swi_blk, axis=0)
    n_rep = np.sum(~np.isnan(rt_rep_blk), axis=0)
    n_swi = np.sum(~np.isnan(rt_swi_blk), axis=0)
    sem_rep_blk = np.nanstd(rt_rep_blk, axis=0, ddof=0) / np.sqrt(n_rep)
    sem_swi_blk = np.nanstd(rt_swi_blk, axis=0, ddof=0) / np.sqrt(n_swi)

    plot_rt_vs_condition(block_levels, mean_rep_blk, mean_swi_blk,
                         sem_rep_blk, sem_swi_blk,
                         xlabel='Block', ylim=(500,900),
                         xmargin=0.5, filename='TOJ_Block_history.pdf')
else:
    print("No block column or empty levels.")