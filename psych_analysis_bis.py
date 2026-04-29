import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import os
from datetime import datetime

# ------------------------------------------------------------------------
# PUBLICATION STYLE SETTINGS (Illustrator‑friendly)
# ------------------------------------------------------------------------
mpl.rcParams['pdf.fonttype'] = 42          # text as text (not outlines)
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'      # or 'Helvetica', 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['lines.linewidth'] = 1.5

# Standard figure widths (inches)
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.0

if __name__ == '__main__':
    # ------------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------------
    output_dir = r"E:\TemporalSwitch\Pymc_psychometric\output_bis"
    processed_csv = os.path.join(output_dir, 'processed_data.csv')
    trace_file = os.path.join(output_dir, 'trace.nc')

    print("=" * 70)
    print("POST‑HOC ANALYSIS – TEMPORAL BISECTION (without lapse/guess)")
    print("=" * 70)
    print(f"Loading processed data: {processed_csv}")
    print(f"Loading trace: {trace_file}")

    # ------------------------------------------------------------------------
    # 1. LOAD DATA AND TRACE
    # ------------------------------------------------------------------------
    df = pd.read_csv(processed_csv)
    idata = az.from_netcdf(trace_file)

    subject_ids = df['subject'].unique()
    n_subjects = len(subject_ids)
    durations = df['duration'].values
    condition_labels = ['Repeat', 'Switch']
    condition_coords = idata.posterior.coords['condition'].values  # ['repeat', 'switch']

    trace = idata.posterior
    posterior_predictive = idata.posterior_predictive

    # Population means
    mu_pse_mean = trace['mu_pse'].mean(dim=['chain', 'draw']).values
    mu_slope_mean = np.exp(trace['mu_log_slope'].mean(dim=['chain', 'draw']).values)
    # Weber fraction (full-width) = (2 * log(3) / slope) / PSE
    weber_mean = (2 * np.log(3) / mu_slope_mean) / mu_pse_mean

    print("\nPopulation parameter estimates (from saved trace):")
    for i, cond in enumerate(condition_coords):
        print(f"  {condition_labels[i]}:")
        print(f"    PSE = {mu_pse_mean[i]:.3f} s")
        print(f"    Slope = {mu_slope_mean[i]:.2f}")
        print(f"    Weber = {weber_mean[i]:.3f}")

    # Unique duration values for tick marks
    dur_unique = np.sort(df['duration'].unique())

    # ------------------------------------------------------------------------
    # 2. PSYCHOMETRIC FUNCTION PLOT (Population Fit) – NO GRID, NO ERROR BARS
    # ------------------------------------------------------------------------
    print("\nGenerating psychometric function plot (population fit)...")
    fig_psych, ax_psych = plt.subplots(1, 1, figsize=(SINGLE_COLUMN, 2.8))
    dur_grid = np.linspace(durations.min(), durations.max(), 100)

    for cond_idx, cond_label in enumerate(condition_coords):
        # Population-level psychometric function (logistic)
        p_pop = stats.logistic.cdf(dur_grid, loc=mu_pse_mean[cond_idx], scale=1/mu_slope_mean[cond_idx])
        line_color = 'black' if cond_idx == 0 else 'gray'
        ax_psych.plot(dur_grid, p_pop, color=line_color, linestyle='-', linewidth=1.5,
                      label=f'{condition_labels[cond_idx]}')

    # Observed data markers (no error bars)
    for cond_idx, cond_label in enumerate(condition_coords):
        cond_data = df[df['hist_code'] == cond_idx]
        obs_stats = cond_data.groupby('duration')['response'].agg(['mean']).reset_index()
        face_color = 'black' if cond_idx == 0 else 'gray'
        edge_color = 'black' if cond_idx == 0 else 'gray'   # Switch edge now gray
        ax_psych.plot(obs_stats['duration'], obs_stats['mean'],
                      'o', color=edge_color, markerfacecolor=face_color,
                      markeredgecolor=edge_color, markersize=5,
                      label='_nolegend_' if cond_idx == 0 else None)

    ax_psych.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_psych.set_xlabel('Duration (s)')
    ax_psych.set_ylabel('Proportion "Long"')
    ax_psych.set_xticks(dur_unique)
    ax_psych.set_xticklabels([f'{x:.1f}' for x in dur_unique])
    ax_psych.legend(loc='best', frameon=False, fontsize=8)
    ax_psych.grid(False)
    ax_psych.spines['top'].set_visible(False)
    ax_psych.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.5)
    psych_png = os.path.join(output_dir, 'psychometric_functions_population.png')
    psych_pdf = os.path.join(output_dir, 'psychometric_functions_population.pdf')
    plt.savefig(psych_png, dpi=300, bbox_inches='tight')
    plt.savefig(psych_pdf, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {psych_png} and {psych_pdf}")

    # ------------------------------------------------------------------------
    # 3. PSYCHOMETRIC FUNCTION PLOT (Model Prediction) – NO GRID, NO ERROR BARS
    # ------------------------------------------------------------------------
    print("\nGenerating psychometric function plot (model prediction)...")
    fig_pred, ax_pred = plt.subplots(1, 1, figsize=(SINGLE_COLUMN, 2.8))

    for cond_idx, cond_label in enumerate(condition_coords):
        # Extract posterior samples for this condition
        pse_samples = trace['pse'].sel(condition=cond_label).values      # (chain, draw, subject)
        slope_samples = trace['slope'].sel(condition=cond_label).values  # (chain, draw, subject)

        n_chains, n_draws, n_subj = pse_samples.shape
        pse_flat = pse_samples.reshape(-1, n_subj)      # (n_samples, n_subj)
        slope_flat = slope_samples.reshape(-1, n_subj)

        pred_mean = np.zeros(len(dur_grid))

        for i, dur_val in enumerate(dur_grid):
            p_subj = stats.logistic.cdf(dur_val, loc=pse_flat, scale=1/slope_flat)
            pred_mean[i] = np.mean(p_subj)   # average over all samples and subjects

        line_color = 'black' if cond_idx == 0 else 'gray'
        ax_pred.plot(dur_grid, pred_mean, color=line_color, linestyle='-', linewidth=1.5,
                     label=f'{condition_labels[cond_idx]} (model)')

    # Observed data markers (no error bars)
    for cond_idx, cond_label in enumerate(condition_coords):
        cond_data = df[df['hist_code'] == cond_idx]
        obs_stats = cond_data.groupby('duration')['response'].agg(['mean']).reset_index()
        face_color = 'black' if cond_idx == 0 else 'gray'
        edge_color = 'black' if cond_idx == 0 else 'gray'   # Switch edge gray
        ax_pred.plot(obs_stats['duration'], obs_stats['mean'],
                     'o', color=edge_color, markerfacecolor=face_color,
                     markeredgecolor=edge_color, markersize=5,
                     label=f'{condition_labels[cond_idx]} (observed)')

    ax_pred.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_pred.set_xlabel('Duration (s)')
    ax_pred.set_ylabel('Proportion "Long"')
    ax_pred.set_xticks(dur_unique)
    ax_pred.set_xticklabels([f'{x:.1f}' for x in dur_unique])
    ax_pred.legend(loc='best', frameon=False, fontsize=8)
    ax_pred.grid(False)
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.5)
    pred_png = os.path.join(output_dir, 'psychometric_functions_model.png')
    pred_pdf = os.path.join(output_dir, 'psychometric_functions_model.pdf')
    plt.savefig(pred_png, dpi=300, bbox_inches='tight')
    plt.savefig(pred_pdf, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {pred_png} and {pred_pdf}")

    # ------------------------------------------------------------------------
    # 4. POSTERIOR PREDICTIVE CHECK – NO GRID, NO ERROR BARS
    # ------------------------------------------------------------------------
    print("\nGenerating posterior predictive check...")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.0))

    dur_grid_fine = np.linspace(durations.min(), durations.max(), 200)

    for cond_idx, cond_label in enumerate(condition_coords):
        ax = axes[cond_idx]
        cond_data = df[df['hist_code'] == cond_idx]

        # Observed proportions (no error bars)
        obs_props = cond_data.groupby('duration')['response'].agg(['mean']).reset_index()

        # Extract posterior samples
        pse_samples = trace['pse'].sel(condition=cond_label).values      # (chain, draw, subject)
        slope_samples = trace['slope'].sel(condition=cond_label).values  # (chain, draw, subject)

        n_chains, n_draws, n_subj = pse_samples.shape
        pse_flat = pse_samples.reshape(-1, n_subj)
        slope_flat = slope_samples.reshape(-1, n_subj)

        # Compute mean predicted probability at each duration grid point
        pred_mean = np.zeros(len(dur_grid_fine))

        for i, dur_val in enumerate(dur_grid_fine):
            p_subj = stats.logistic.cdf(dur_val, loc=pse_flat, scale=1/slope_flat)
            pred_mean[i] = np.mean(p_subj)

        # Population fit using population means
        p_pop = stats.logistic.cdf(dur_grid_fine, loc=mu_pse_mean[cond_idx], scale=1/mu_slope_mean[cond_idx])

        # Observed markers
        face_color = 'black' if cond_idx == 0 else 'gray'
        edge_color = 'black' if cond_idx == 0 else 'gray'
        ax.plot(obs_props['duration'], obs_props['mean'],
                'o', color=edge_color, markerfacecolor=face_color,
                markeredgecolor=edge_color, markersize=4,
                label='Observed', zorder=10)

        # Model prediction (black or gray solid)
        line_color = 'black' if cond_idx == 0 else 'gray'
        ax.plot(dur_grid_fine, pred_mean, color=line_color, linewidth=1.5, label='Model prediction')
        # Population fit (always gray dashed, for reference)
        ax.plot(dur_grid_fine, p_pop, color='gray', linestyle='--',
                linewidth=1, alpha=0.7, label='Population fit')

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axvline(mu_pse_mean[cond_idx], color='black', linestyle=':', alpha=0.7, linewidth=0.8)

        ax.set_xticks(dur_unique)
        ax.set_xticklabels([f'{x:.1f}' for x in dur_unique])
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Proportion "Long"')
        ax.set_title(f'{condition_labels[cond_idx]}', fontsize=10)
        ax.legend(loc='best', fontsize=7, frameon=False)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=1.0)
    ppc_png = os.path.join(output_dir, 'posterior_predictive.png')
    ppc_pdf = os.path.join(output_dir, 'posterior_predictive.pdf')
    plt.savefig(ppc_png, dpi=300, bbox_inches='tight')
    plt.savefig(ppc_pdf, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {ppc_png} and {ppc_pdf}")

    # ------------------------------------------------------------------------
    # 5. SUMMARY TABLES (with 95% HDI) – unchanged
    # ------------------------------------------------------------------------
    print("\nGenerating summary tables...")
    summary_data = []
    for cond_idx, cond_label in enumerate(condition_coords):
        # PSE
        pse_samps = trace['mu_pse'].sel(condition=cond_label).values.flatten()
        pse_m = np.mean(pse_samps)
        pse_h = az.hdi(pse_samps, hdi_prob=0.95)
        # Slope
        slope_samps = np.exp(trace['mu_log_slope'].sel(condition=cond_label).values.flatten())
        slope_m = np.mean(slope_samps)
        slope_h = az.hdi(slope_samps, hdi_prob=0.95)
        # Weber fraction (full-width)
        weber_samps = (2 * np.log(3) / slope_samps) / pse_samps
        weber_m = np.mean(weber_samps)
        weber_h = az.hdi(weber_samps, hdi_prob=0.95)

        summary_data.append({
            'Condition': condition_labels[cond_idx],
            'PSE (s)': f"{pse_m:.3f} [{pse_h[0]:.3f}, {pse_h[1]:.3f}]",
            'Slope': f"{slope_m:.2f} [{slope_h[0]:.2f}, {slope_h[1]:.2f}]",
            'Weber': f"{weber_m:.3f} [{weber_h[0]:.3f}, {weber_h[1]:.3f}]"
        })

    # Differences (Switch - Repeat)
    def diff_summary(samps1, samps2):
        diff = samps1 - samps2
        m = np.mean(diff)
        h = az.hdi(diff, hdi_prob=0.95)
        pd = np.mean(diff > 0) * 100 if np.median(diff) > 0 else np.mean(diff < 0) * 100
        return f"{m:.3f} [{h[0]:.3f}, {h[1]:.3f}]", pd

    # Compute difference samples (Switch - Repeat)
    pse_diff_vals = trace['mu_pse'].sel(condition=condition_coords[1]).values.flatten() - trace['mu_pse'].sel(condition=condition_coords[0]).values.flatten()
    slope_diff_vals = np.exp(trace['mu_log_slope'].sel(condition=condition_coords[1]).values.flatten()) - np.exp(trace['mu_log_slope'].sel(condition=condition_coords[0]).values.flatten())
    # Weber difference – compute from population samples with full-width
    weber0_samps = (2 * np.log(3) / np.exp(trace['mu_log_slope'].sel(condition=condition_coords[0]).values.flatten())) / trace['mu_pse'].sel(condition=condition_coords[0]).values.flatten()
    weber1_samps = (2 * np.log(3) / np.exp(trace['mu_log_slope'].sel(condition=condition_coords[1]).values.flatten())) / trace['mu_pse'].sel(condition=condition_coords[1]).values.flatten()
    weber_diff_vals = weber1_samps - weber0_samps

    pse_diff_str, pse_pd = diff_summary(
        trace['mu_pse'].sel(condition=condition_coords[1]).values.flatten(),
        trace['mu_pse'].sel(condition=condition_coords[0]).values.flatten()
    )
    slope_diff_str, slope_pd = diff_summary(
        np.exp(trace['mu_log_slope'].sel(condition=condition_coords[1]).values.flatten()),
        np.exp(trace['mu_log_slope'].sel(condition=condition_coords[0]).values.flatten())
    )
    weber_diff_str, weber_pd = diff_summary(weber1_samps, weber0_samps)

    summary_data.append({
        'Condition': 'Difference (Switch - Repeat)',
        'PSE (s)': pse_diff_str,
        'Slope': slope_diff_str,
        'Weber': weber_diff_str
    })

    summary_df = pd.DataFrame(summary_data)
    print("\nParameter estimates by condition (mean [95% HDI]):")
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(output_dir, 'summary_table.csv')
    summary_txt = os.path.join(output_dir, 'summary_table.txt')
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(summary_df.to_string(index=False))
    print(f"  Saved: {summary_csv} and {summary_txt}")

    # ------------------------------------------------------------------------
    # 6. SAVE DETAILED PARAMETER TABLES (individual subjects) – unchanged
    # ------------------------------------------------------------------------
    subject_rows = []
    for subj in subject_ids:
        for cond_idx, cond_label in enumerate(condition_coords):
            # PSE
            pse_m = trace['pse'].sel(subject=subj, condition=cond_label).mean().values
            pse_h = az.hdi(trace['pse'].sel(subject=subj, condition=cond_label).values.flatten(), hdi_prob=0.95)
            # Slope
            slope_m = trace['slope'].sel(subject=subj, condition=cond_label).mean().values
            slope_h = az.hdi(trace['slope'].sel(subject=subj, condition=cond_label).values.flatten(), hdi_prob=0.95)
            # Weber fraction (full-width) – subject‑specific
            weber_vals = (2 * np.log(3) / trace['slope'].sel(subject=subj, condition=cond_label).values.flatten()) / trace['pse'].sel(subject=subj, condition=cond_label).values.flatten()
            weber_m = np.mean(weber_vals)
            weber_h = az.hdi(weber_vals, hdi_prob=0.95)

            subject_rows.append({
                'subject': subj,
                'condition': condition_labels[cond_idx],
                'pse_mean': pse_m,
                'pse_hdi_lower': pse_h[0],
                'pse_hdi_upper': pse_h[1],
                'slope_mean': slope_m,
                'slope_hdi_lower': slope_h[0],
                'slope_hdi_upper': slope_h[1],
                'weber_mean': weber_m,
                'weber_hdi_lower': weber_h[0],
                'weber_hdi_upper': weber_h[1]
            })
    subject_df = pd.DataFrame(subject_rows)
    subject_csv = os.path.join(output_dir, 'individual_results.csv')
    subject_df.to_csv(subject_csv, index=False)
    print(f"  Saved: {subject_csv}")

    # Population results (already computed above, but save as CSV) – unchanged
    pop_rows = []
    for cond_idx, cond_label in enumerate(condition_coords):
        # PSE
        pse_pop = trace['mu_pse'].sel(condition=cond_label).values.flatten()
        pse_h = az.hdi(pse_pop, hdi_prob=0.95)
        # Slope
        slope_pop = np.exp(trace['mu_log_slope'].sel(condition=cond_label).values.flatten())
        slope_h = az.hdi(slope_pop, hdi_prob=0.95)
        # Weber (full-width)
        weber_pop = (2 * np.log(3) / slope_pop) / pse_pop
        weber_h = az.hdi(weber_pop, hdi_prob=0.95)

        pop_rows.append({
            'condition': condition_labels[cond_idx],
            'pse_mean': np.mean(pse_pop),
            'pse_hdi_lower': pse_h[0],
            'pse_hdi_upper': pse_h[1],
            'slope_mean': np.mean(slope_pop),
            'slope_hdi_lower': slope_h[0],
            'slope_hdi_upper': slope_h[1],
            'weber_mean': np.mean(weber_pop),
            'weber_hdi_lower': weber_h[0],
            'weber_hdi_upper': weber_h[1]
        })
    pop_df = pd.DataFrame(pop_rows)
    pop_csv = os.path.join(output_dir, 'population_results.csv')
    pop_df.to_csv(pop_csv, index=False)
    print(f"  Saved: {pop_csv}")

    # Differences CSV (Switch - Repeat) – unchanged
    diff_df = pd.DataFrame({
        'parameter': ['PSE', 'Slope', 'Weber'],
        'mean': [np.mean(pse_diff_vals), np.mean(slope_diff_vals), np.mean(weber_diff_vals)],
        'hdi_lower': [az.hdi(pse_diff_vals, hdi_prob=0.95)[0],
                      az.hdi(slope_diff_vals, hdi_prob=0.95)[0],
                      az.hdi(weber_diff_vals, hdi_prob=0.95)[0]],
        'hdi_upper': [az.hdi(pse_diff_vals, hdi_prob=0.95)[1],
                      az.hdi(slope_diff_vals, hdi_prob=0.95)[1],
                      az.hdi(weber_diff_vals, hdi_prob=0.95)[1]],
        'prob_direction': [pse_pd, slope_pd, weber_pd]
    })
    diff_csv = os.path.join(output_dir, 'differences.csv')
    diff_df.to_csv(diff_csv, index=False)
    print(f"  Saved: {diff_csv}")

    # ------------------------------------------------------------------------
    # 7. FINAL INTERPRETATION – unchanged
    # ------------------------------------------------------------------------
    print("\nGenerating final summary text...")
    final_txt = os.path.join(output_dir, 'final_summary.txt')
    with open(final_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TEMPORAL BISECTION ANALYSIS - FINAL SUMMARY (without lapse/guess)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {processed_csv}\n")
        f.write(f"Trace file: {trace_file}\n")
        f.write(f"Number of subjects: {n_subjects}\n\n")

        f.write("POPULATION PARAMETER ESTIMATES (mean [95% HDI]):\n")
        f.write("-" * 60 + "\n")
        for cond_idx, cond_label in enumerate(condition_coords):
            pse_h = az.hdi(trace['mu_pse'].sel(condition=cond_label).values.flatten(), hdi_prob=0.95)
            slope_h = az.hdi(np.exp(trace['mu_log_slope'].sel(condition=cond_label).values.flatten()), hdi_prob=0.95)
            weber_vals_full = (2 * np.log(3) / np.exp(trace['mu_log_slope'].sel(condition=cond_label).values.flatten())) / trace['mu_pse'].sel(condition=cond_label).values.flatten()
            weber_h = az.hdi(weber_vals_full, hdi_prob=0.95)
            f.write(f"\n{condition_labels[cond_idx]}:\n")
            f.write(f"  PSE:    {mu_pse_mean[cond_idx]:.3f} s [{pse_h[0]:.3f}, {pse_h[1]:.3f}]\n")
            f.write(f"  Slope:  {mu_slope_mean[cond_idx]:.2f} [{slope_h[0]:.2f}, {slope_h[1]:.2f}]\n")
            f.write(f"  Weber:  {weber_mean[cond_idx]:.3f} [{weber_h[0]:.3f}, {weber_h[1]:.3f}]\n")

        f.write("\n\nCONDITION DIFFERENCES (Switch - Repeat):\n")
        f.write("-" * 60 + "\n")
        f.write(f"PSE:   {pse_diff_str}   PD = {pse_pd:.1f}%\n")
        f.write(f"Slope: {slope_diff_str} PD = {slope_pd:.1f}%\n")
        f.write(f"Weber: {weber_diff_str} PD = {weber_pd:.1f}%\n")

        f.write("\n\nINTERPRETATION:\n")
        f.write("-" * 60 + "\n")
        # Use the difference variables (Switch - Repeat)
        pse_diff_hdi = az.hdi(pse_diff_vals, hdi_prob=0.95)
        weber_diff_hdi = az.hdi(weber_diff_vals, hdi_prob=0.95)

        if pse_diff_hdi[0] > 0:
            f.write(f"• Switch history shifts PSE to the right by {np.mean(pse_diff_vals):.3f} s\n")
        elif pse_diff_hdi[1] < 0:
            f.write(f"• Repeat history shifts PSE to the right by {abs(np.mean(pse_diff_vals)):.3f} s\n")
        else:
            f.write(f"• No credible difference in PSE between conditions\n")

        if weber_diff_hdi[0] > 0:
            f.write(f"• Switch history shows higher Weber fraction (lower sensitivity) by {np.mean(weber_diff_vals):.3f}\n")
        elif weber_diff_hdi[1] < 0:
            f.write(f"• Repeat history shows higher Weber fraction (lower sensitivity) by {abs(np.mean(weber_diff_vals)):.3f}\n")
        else:
            f.write(f"• No credible difference in Weber fraction (sensitivity) between conditions\n")

    print(f"  Saved: {final_txt}")

    print("\n" + "=" * 80)
    print("POST‑HOC ANALYSIS COMPLETE (without lapse/guess)")
    print("=" * 80)