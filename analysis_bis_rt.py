import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

if __name__ == '__main__':
    # ------------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------------
    base_dir = r"E:\TemporalSwitch\Pymc_reaction_time"
    output_dir = os.path.join(base_dir, "output_rt_bis_centered")
    processed_csv = os.path.join(output_dir, 'rt_processed_data.csv')
    trace_file = os.path.join(output_dir, 'rt_trace.nc')
    center_json = os.path.join(output_dir, 'centering_params.json')

    print("=" * 70)
    print("POSTERIOR PREDICTIVE CHECK – REACTION TIME (MEAN‑CENTERED PREDICTORS)")
    print("=" * 70)
    print(f"Loading processed data: {processed_csv}")
    print(f"Loading trace: {trace_file}")

    # Load centering parameters
    with open(center_json, 'r') as f:
        center = json.load(f)
    duration_mean = center['duration_mean']   # in 100 ms units
    rsi_mean = center['rsi_mean']
    block_mean = center['block_mean']

    # ------------------------------------------------------------------------
    # 1. LOAD DATA AND TRACE
    # ------------------------------------------------------------------------
    df = pd.read_csv(processed_csv)
    idata = az.from_netcdf(trace_file)

    condition_labels = ['Repeat', 'Switch']

    # Original duration values (seconds) for plotting
    dur_original = np.sort(df['duration'].unique())  # seconds
    # Convert to 100 ms units and then center
    dur_100ms = dur_original * 10
    dur_c = dur_100ms - duration_mean

    trace = idata.posterior

    # ------------------------------------------------------------------------
    # 2. POSTERIOR PREDICTIVE CHECK
    # ------------------------------------------------------------------------
    print("\nGenerating posterior predictive plot...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Extract fixed‑effect samples as flat arrays
    mu_intercept = trace['mu_intercept'].values.flatten()
    mu_hist = trace['mu_hist'].values.flatten()
    mu_dur = trace['mu_dur'].values.flatten()
    mu_dur2 = trace['mu_dur2'].values.flatten()
    mu_block = trace['mu_block'].values.flatten()
    mu_rsi = trace['mu_rsi'].values.flatten()
    mu_hist_dur = trace['mu_hist_dur'].values.flatten()
    mu_hist_dur2 = trace['mu_hist_dur2'].values.flatten()
    mu_hist_block = trace['mu_hist_block'].values.flatten()
    mu_hist_rsi = trace['mu_hist_rsi'].values.flatten()

    n_draws = len(mu_intercept)

    for cond in [0, 1]:
        ax = axes[cond]
        cond_mask = df['hist_code'] == cond
        obs_dur = df.loc[cond_mask, 'duration']  # original seconds
        obs_rt = df.loc[cond_mask, 'rt_ms']

        # Observed mean and 95% CI per duration
        obs_stats = pd.DataFrame({'duration': obs_dur, 'rt_ms': obs_rt}).groupby('duration')['rt_ms'].agg(['mean', 'sem']).reset_index()
        obs_stats['lower'] = obs_stats['mean'] - 1.96 * obs_stats['sem']
        obs_stats['upper'] = obs_stats['mean'] + 1.96 * obs_stats['sem']

        # Predictions on grid (average block and rsi, i.e., block_c=0, rsi_c=0)
        pred_rt = np.zeros((len(dur_c), n_draws))
        for i, dc in enumerate(dur_c):
            pred_rt[i, :] = (mu_intercept +
                             mu_hist * cond +
                             mu_dur * dc +
                             mu_dur2 * dc**2 +
                             mu_block * 0 +
                             mu_rsi * 0 +
                             mu_hist_dur * cond * dc +
                             mu_hist_dur2 * cond * dc**2 +
                             mu_hist_block * cond * 0 +
                             mu_hist_rsi * cond * 0)
        pred_mean = pred_rt.mean(axis=1)
        pred_lower = np.percentile(pred_rt, 2.5, axis=1)
        pred_upper = np.percentile(pred_rt, 97.5, axis=1)

        ax.errorbar(obs_stats['duration'], obs_stats['mean'],
                    yerr=[obs_stats['mean'] - obs_stats['lower'], obs_stats['upper'] - obs_stats['mean']],
                    fmt='o', color='black', capsize=3, label='Observed ± 95% CI')
        ax.plot(dur_original, pred_mean, color='red', linewidth=3, label='Model prediction')
        ax.fill_between(dur_original, pred_lower, pred_upper, color='lightcoral', alpha=0.5, label='95% HDI')
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Reaction Time (ms)')
        ax.set_title(f'Condition {condition_labels[cond]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ppc_png = os.path.join(output_dir, 'rt_posterior_predictive.png')
    ppc_pdf = os.path.join(output_dir, 'rt_posterior_predictive.pdf')
    plt.savefig(ppc_png, dpi=300, bbox_inches='tight')
    plt.savefig(ppc_pdf, bbox_inches='tight')
    plt.show()
    print(f"Posterior predictive plot saved: {ppc_png} and {ppc_pdf}")

    # ------------------------------------------------------------------------
    # 3. FIXED‑EFFECTS SUMMARY TABLE
    # ------------------------------------------------------------------------
    print("\nGenerating fixed‑effects summary table...")
    var_names = [
        'mu_intercept', 'mu_hist', 'mu_dur', 'mu_dur2',
        'mu_block', 'mu_rsi', 'mu_hist_dur', 'mu_hist_dur2',
        'mu_hist_block', 'mu_hist_rsi', 'sigma'
    ]
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95, round_to=3)
    summary_csv = os.path.join(output_dir, 'rt_fixed_effects_summary.csv')
    summary.to_csv(summary_csv)
    print("\nFixed‑effects summary (mean, sd, HDI, mcse, ess, r_hat):")
    print(summary.to_string())
    print(f"Saved to: {summary_csv}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)