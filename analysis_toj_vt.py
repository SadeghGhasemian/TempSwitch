import warnings
warnings.filterwarnings("ignore")

import hddm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from kabuki.analyze import gelman_rubin
from kabuki.utils import concat_models

# ============================================
# Configuration
# ============================================
base_dir = r'E:\TemporalSwitch\HDDM_latest\power_analysis_toj_vt'
models_base = os.path.join(base_dir, 'downsampling_models')
results_base = os.path.join(base_dir, 'downsampling_results')
os.makedirs(results_base, exist_ok=True)

sample_sizes = [10, 15, 20, 25, 30, 35]
n_chains = 4

params_of_interest = [
    'v_Intercept',
    "v_C(hist, Treatment('R'))[T.S]",
    'v_soa',
    "v_C(hist, Treatment('R'))[T.S]:soa"
    't_Intercept',
    "t_C(hist, Treatment('R'))[T.S]",
    't_soa',
    "t_C(hist, Treatment('R'))[T.S]:soa"
]

# ============================================
# Process each sample size
# ============================================
combined_traces = {}

for n in sample_sizes:
    print(f"\nProcessing N = {n}")
    n_dir = os.path.join(results_base, f'N{n}')
    os.makedirs(n_dir, exist_ok=True)

    # Load the 4 chains
    chains = []
    for ch in range(n_chains):
        fname = os.path.join(models_base, f'N{n}', f'chain{ch}')
        with open(fname, 'rb') as f:
            chains.append(pickle.load(f))

    # Concatenate chains
    model_comb = concat_models(chains)

    # Save traces
    traces = model_comb.get_traces()
    traces_file = os.path.join(n_dir, 'traces.csv')
    traces.to_csv(traces_file)

    # Print column names to debug
    print("Trace columns:", list(traces.columns))

    # Save statistics
    stats = model_comb.gen_stats()
    stats_file = os.path.join(n_dir, 'stats.csv')
    stats.to_csv(stats_file)

    # Save DIC
    dic_file = os.path.join(n_dir, 'DIC.csv')
    pd.DataFrame({'DIC': [model_comb.dic]}).to_csv(dic_file, index=False)

    # Compute and save R‑hat
    rhat = gelman_rubin(chains)
    rhat_file = os.path.join(n_dir, 'rhat.csv')
    pd.DataFrame.from_dict(rhat, orient='index').to_csv(rhat_file)

    # Save posterior plots
    plot_dir = os.path.join(n_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    current_dir = os.getcwd()
    os.chdir(plot_dir)
    model_comb.plot_posteriors(save=True)
    os.chdir(current_dir)

    # Store traces for violin plots - ensure columns exist
    available_cols = [col for col in params_of_interest if col in traces.columns]
    if len(available_cols) < len(params_of_interest):
        missing = set(params_of_interest) - set(available_cols)
        print(f"Warning: Missing columns: {missing}. Using available.")
    combined_traces[n] = traces[available_cols]

# Summary table
summary_rows = []
for n in sample_sizes:
    row = {'N': n}
    for param in params_of_interest:
        if n in combined_traces and param in combined_traces[n].columns:
            vals = combined_traces[n][param].values
            row[f'{param}_mean'] = np.mean(vals)
            row[f'{param}_std'] = np.std(vals)
            row[f'{param}_ci_low'] = np.percentile(vals, 2.5)
            row[f'{param}_ci_high'] = np.percentile(vals, 97.5)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(results_base, 'summary.csv'), index=False)

print("\nAll done.")

a = np.array([0.1, 0.072, 0.044, 0.016])
b = np.mean(a)
c = a-b