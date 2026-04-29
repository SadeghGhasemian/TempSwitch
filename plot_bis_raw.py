"""
Bisection RT plots (RSI × history, Duration × history, Block × history)
Corrected x-axis margin for block plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------- Configuration --------------------
base_dir = Path('E:/TemporalSwitch/Pymc_reaction_time')
data_file = base_dir / 'bis_rt.csv'
output_dir = base_dir / 'output_bis_raw'
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.2)

# -------------------- Load and prepare data --------------------
data = pd.read_csv(data_file)
data.columns = data.columns.str.lower()

print("Column names:", data.columns.tolist())
print("First 5 rows:\n", data.head())

# Convert to ms if needed – assume seconds if max < 100
if data['rsi'].max() < 100:
    print("Converting rsi from seconds to ms")
    data['rsi'] = data['rsi'] * 1000
if 'duration' in data and data['duration'].max() < 100:
    print("Converting duration from seconds to ms")
    data['duration'] = data['duration'] * 1000

# History levels
hist_vals = data['hist'].unique()
print("Unique history values:", hist_vals)
if len(hist_vals) != 2:
    raise ValueError(f"Expected 2 history levels, got {len(hist_vals)}: {hist_vals}")

# Map to standard names 'repeat' and 'switch'
data['hist'] = pd.Categorical(data['hist'], categories=hist_vals, ordered=True)
hist_map = {hist_vals[0]: 'repeat', hist_vals[1]: 'switch'}
data['hist'] = data['hist'].cat.rename_categories(hist_map)
print("Mapped history to:", data['hist'].cat.categories.tolist())

subjects = data['subject'].unique()
print(f"Number of subjects: {len(subjects)}")

rsi_levels = np.sort(data['rsi'].unique())
duration_levels = np.sort(data['duration'].unique()) if 'duration' in data else np.array([])
block_levels = np.sort(data['block'].unique()) if 'block' in data else np.array([])

print("RSI levels:", rsi_levels)
if 'duration' in data:
    print("Duration levels:", duration_levels)
if 'block' in data:
    print("Block levels:", block_levels)

hist_levels = data['hist'].cat.categories  # ['repeat', 'switch']

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
                    rt_array[i, j, k] = subset.mean() * 1000   # convert rt to ms (if in seconds)
    return rt_array

# -------------------- Plotting function --------------------
def plot_rt_vs_condition(condition_values, rt_repeat, rt_switch,
                          sem_repeat, sem_switch,
                          xlabel, ylim, xtick_rotation=0,
                          xmargin=50,  # default margin for RSI/duration
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
                         xlabel='RSI (ms)', ylim=(300,700), xtick_rotation=45,
                         xmargin=50, filename='BIS_RSI_history.pdf')
else:
    print("No RSI data found.")

# -------------------- Duration plot --------------------
if 'duration' in data and len(duration_levels) > 0:
    rt_array_dur = compute_subject_condition_means(data, 'duration', duration_levels)
    print("rt_array_dur shape:", rt_array_dur.shape)
    print("Non-NaN count:", np.sum(~np.isnan(rt_array_dur)))

    rt_rep_dur = rt_array_dur[:, :, 0]
    rt_swi_dur = rt_array_dur[:, :, 1]

    mean_rep_dur = np.nanmean(rt_rep_dur, axis=0)
    mean_swi_dur = np.nanmean(rt_swi_dur, axis=0)
    n_rep = np.sum(~np.isnan(rt_rep_dur), axis=0)
    n_swi = np.sum(~np.isnan(rt_swi_dur), axis=0)
    sem_rep_dur = np.nanstd(rt_rep_dur, axis=0, ddof=0) / np.sqrt(n_rep)
    sem_swi_dur = np.nanstd(rt_swi_dur, axis=0, ddof=0) / np.sqrt(n_swi)

    plot_rt_vs_condition(duration_levels, mean_rep_dur, mean_swi_dur,
                         sem_rep_dur, sem_swi_dur,
                         xlabel='Duration (ms)', ylim=(300,700),
                         xmargin=50, filename='BIS_Duration_history.pdf')
else:
    print("No duration column or empty levels.")

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
                         xlabel='Block', ylim=(300,700),
                         xmargin=0.5, filename='BIS_Block_history.pdf')
else:
    print("No block column or empty levels.")