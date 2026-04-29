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
    # CONFIGURATION – FIXED FILENAMES
    # ------------------------------------------------------------------------
    output_dir = r"E:\TemporalSwitch\Pymc_psychometric\output_toj"
    processed_csv = os.path.join(output_dir, 'processed_data.csv')
    trace_file = os.path.join(output_dir, 'trace.nc')

    print("=" * 70)
    print("POST‑HOC ANALYSIS (without lapse/guess)")
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
    soas = df['soa'].values
    condition_labels = ['Repeat', 'Switch']

    condition_coords = idata.posterior.coords['condition'].values  # e.g. ['repeat', 'switch']

    trace = idata.posterior
    posterior_predictive = idata.posterior_predictive

    mu_pss_mean = trace['mu_pss'].mean(dim=['chain', 'draw']).values
    mu_slope_mean = np.exp(trace['mu_slope_log'].mean(dim=['chain', 'draw']).values)
    jnd_mean = 1.35 / mu_slope_mean

    print("\nPopulation parameter estimates (from saved trace):")
    for i, cond in enumerate(condition_coords):
        print(f"  {condition_labels[i]}:")
        print(f"    PSS = {mu_pss_mean[i]:.2f} ms")
        print(f"    Slope = {mu_slope_mean[i]:.4f}")
        print(f"    JND = {jnd_mean[i]:.2f} ms")

    soa_unique = np.sort(df['soa'].unique())

    # ------------------------------------------------------------------------
    # 2. PSYCHOMETRIC FUNCTION PLOT (Population Fit) – NO GRID
    # ------------------------------------------------------------------------
    print("\nGenerating psychometric function plot (population fit)...")
    fig_psych, ax_psych = plt.subplots(1, 1, figsize=(SINGLE_COLUMN, 2.8))

    soa_grid = np.linspace(soas.min(), soas.max(), 100)

    for cond_idx, cond_label in enumerate(condition_coords):
        p_pop = stats.norm.cdf(soa_grid, loc=mu_pss_mean[cond_idx], scale=1/mu_slope_mean[cond_idx])
        line_color = 'black' if cond_idx == 0 else 'gray'
        ax_psych.plot(soa_grid, p_pop, color=line_color, linestyle='-', linewidth=1.5,
                      label=f'{condition_labels[cond_idx]}')

    # Observed data markers (no error bars)
    for cond_idx, cond_label in enumerate(condition_coords):
        cond_data = df[df['hist_code'] == cond_idx]
        obs_stats = cond_data.groupby('soa')['response'].agg(['mean']).reset_index()
        face_color = 'black' if cond_idx == 0 else 'gray'
        edge_color = 'black' if cond_idx == 0 else 'gray'   # Switch edge now gray
        ax_psych.plot(obs_stats['soa'], obs_stats['mean'],
                      'o', color=edge_color, markerfacecolor=face_color,
                      markeredgecolor=edge_color, markersize=5,
                      label='_nolegend_' if cond_idx == 0 else None)

    ax_psych.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_psych.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax_psych.set_xlabel('SOA (ms)')
    ax_psych.set_ylabel('P(respond "Right")')
    ax_psych.set_xticks(soa_unique)
    ax_psych.set_xticklabels([f'{int(x)}' for x in soa_unique])
    ax_psych.legend(loc='best', frameon=False, fontsize=8)
    ax_psych.grid(False)                     # NO GRID
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
    # 3. PSYCHOMETRIC FUNCTION PLOT (Model Prediction) – NO GRID
    # ------------------------------------------------------------------------
    print("\nGenerating psychometric function plot (model prediction)...")
    fig_pred, ax_pred = plt.subplots(1, 1, figsize=(SINGLE_COLUMN, 2.8))

    for cond_idx, cond_label in enumerate(condition_coords):
        pss_samples = trace['pss'].sel(condition=cond_label).values      # (chain, draw, subject)
        slope_samples = trace['slope'].sel(condition=cond_label).values  # (chain, draw, subject)

        n_chains, n_draws, n_subj = pss_samples.shape
        pss_flat = pss_samples.reshape(-1, n_subj)
        slope_flat = slope_samples.reshape(-1, n_subj)

        pred_mean = np.zeros(len(soa_grid))
        for i, soa_val in enumerate(soa_grid):
            p_subj = stats.norm.cdf(soa_val, loc=pss_flat, scale=1/slope_flat)
            pred_mean[i] = np.mean(p_subj)

        line_color = 'black' if cond_idx == 0 else 'gray'
        ax_pred.plot(soa_grid, pred_mean, color=line_color, linestyle='-', linewidth=1.5,
                     label=f'{condition_labels[cond_idx]} (model)')

    # Observed data markers (no error bars)
    for cond_idx, cond_label in enumerate(condition_coords):
        cond_data = df[df['hist_code'] == cond_idx]
        obs_stats = cond_data.groupby('soa')['response'].agg(['mean']).reset_index()
        face_color = 'black' if cond_idx == 0 else 'gray'
        edge_color = 'black' if cond_idx == 0 else 'gray'   # Switch edge now gray
        ax_pred.plot(obs_stats['soa'], obs_stats['mean'],
                     'o', color=edge_color, markerfacecolor=face_color,
                     markeredgecolor=edge_color, markersize=5,
                     label=f'{condition_labels[cond_idx]} (observed)')

    ax_pred.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_pred.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax_pred.set_xlabel('SOA (ms)')
    ax_pred.set_ylabel('P(respond "Right")')
    ax_pred.set_xticks(soa_unique)
    ax_pred.set_xticklabels([f'{int(x)}' for x in soa_unique])
    ax_pred.legend(loc='best', frameon=False, fontsize=8)
    ax_pred.grid(False)                     # NO GRID
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
    # 4. POSTERIOR PREDICTIVE CHECK (without lapse/guess) – NO GRID
    # ------------------------------------------------------------------------
    print("\nGenerating posterior predictive check...")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.0))

    soa_grid = np.linspace(soas.min(), soas.max(), 200)

    for cond_idx, cond_label in enumerate(condition_coords):
        ax = axes[cond_idx]
        cond_data = df[df['hist_code'] == cond_idx]

        obs_props = cond_data.groupby('soa')['response'].agg(['mean', 'count']).reset_index()

        pss_samples = trace['pss'].sel(condition=cond_label).values
        slope_samples = trace['slope'].sel(condition=cond_label).values

        n_chains, n_draws, n_subj = pss_samples.shape
        pss_flat = pss_samples.reshape(-1, n_subj)
        slope_flat = slope_samples.reshape(-1, n_subj)

        pred_mean = np.zeros(len(soa_grid))
        for i, soa_val in enumerate(soa_grid):
            p_subj = stats.norm.cdf(soa_val, loc=pss_flat, scale=1/slope_flat)
            pred_mean[i] = np.mean(p_subj)

        p_pop = stats.norm.cdf(soa_grid, loc=mu_pss_mean[cond_idx], scale=1/mu_slope_mean[cond_idx])

        # Observed data markers (no error bars) – edge matches face
        face_color = 'black' if cond_idx == 0 else 'gray'
        edge_color = 'black' if cond_idx == 0 else 'gray'
        ax.plot(obs_props['soa'], obs_props['mean'],
                'o', color=edge_color, markerfacecolor=face_color,
                markeredgecolor=edge_color, markersize=4,
                label='Observed', zorder=10)

        # Model prediction (gray or black solid)
        line_color = 'black' if cond_idx == 0 else 'gray'
        ax.plot(soa_grid, pred_mean, color=line_color, linewidth=1.5, label='Model prediction')
        # Population fit (always gray dashed, for reference)
        ax.plot(soa_grid, p_pop, color='gray', linestyle='--',
                linewidth=1, alpha=0.7, label='Population fit')

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axvline(mu_pss_mean[cond_idx], color='black', linestyle=':', alpha=0.7, linewidth=0.8)

        ax.set_xticks(soa_unique)
        ax.set_xticklabels([f'{int(x)}' for x in soa_unique])
        ax.set_xlabel('SOA (ms)')
        ax.set_ylabel('Proportion "Right" Responses')
        ax.set_title(f'{condition_labels[cond_idx]}', fontsize=10)
        ax.legend(loc='best', fontsize=7, frameon=False)
        ax.grid(False)                     # NO GRID
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
    # 5. SUMMARY TABLES (unchanged)
    # ------------------------------------------------------------------------
    # (All your existing table generation code goes here – no changes needed)
    # For brevity, I have kept it identical to your original.
    summary_data = []
    for cond_idx, cond_label in enumerate(condition_coords):
        # PSS
        pss_samps = trace['mu_pss'].sel(condition=cond_label).values.flatten()
        pss_m = np.mean(pss_samps)
        pss_h = az.hdi(pss_samps, hdi_prob=0.95)
        # Slope
        slope_samps = np.exp(trace['mu_slope_log'].sel(condition=cond_label).values.flatten())
        slope_m = np.mean(slope_samps)
        slope_h = az.hdi(slope_samps, hdi_prob=0.95)
        # JND
        jnd_samps = trace['threshold'].sel(condition=cond_label).values.flatten()
        jnd_m = np.mean(jnd_samps)
        jnd_h = az.hdi(jnd_samps, hdi_prob=0.95)

        summary_data.append({
            'Condition': condition_labels[cond_idx],
            'PSS (ms)': f"{pss_m:.2f} [{pss_h[0]:.2f}, {pss_h[1]:.2f}]",
            'Slope': f"{slope_m:.4f} [{slope_h[0]:.4f}, {slope_h[1]:.4f}]",
            'JND (ms)': f"{jnd_m:.2f} [{jnd_h[0]:.2f}, {jnd_h[1]:.2f}]"
        })

    # Differences (Switch - Repeat)
    def diff_summary(samps1, samps2):
        diff = samps1 - samps2
        m = np.mean(diff)
        h = az.hdi(diff, hdi_prob=0.95)
        pd = np.mean(diff > 0) * 100 if np.median(diff) > 0 else np.mean(diff < 0) * 100
        return f"{m:.3f} [{h[0]:.3f}, {h[1]:.3f}]", pd

    pss_diff_vals = (trace['mu_pss'].sel(condition=condition_coords[1]).values.flatten() -
                     trace['mu_pss'].sel(condition=condition_coords[0]).values.flatten())
    slope_diff_vals = (np.exp(trace['mu_slope_log'].sel(condition=condition_coords[1]).values.flatten()) -
                       np.exp(trace['mu_slope_log'].sel(condition=condition_coords[0]).values.flatten()))
    jnd_diff_vals = (trace['threshold'].sel(condition=condition_coords[1]).values.flatten() -
                     trace['threshold'].sel(condition=condition_coords[0]).values.flatten())

    pss_diff_str, pss_pd = diff_summary(
        trace['mu_pss'].sel(condition=condition_coords[1]).values.flatten(),
        trace['mu_pss'].sel(condition=condition_coords[0]).values.flatten()
    )
    slope_diff_str, slope_pd = diff_summary(
        np.exp(trace['mu_slope_log'].sel(condition=condition_coords[1]).values.flatten()),
        np.exp(trace['mu_slope_log'].sel(condition=condition_coords[0]).values.flatten())
    )
    jnd_diff_str, jnd_pd = diff_summary(
        trace['threshold'].sel(condition=condition_coords[1]).values.flatten(),
        trace['threshold'].sel(condition=condition_coords[0]).values.flatten()
    )

    summary_data.append({
        'Condition': 'Difference (Switch - Repeat)',
        'PSS (ms)': pss_diff_str,
        'Slope': slope_diff_str,
        'JND (ms)': jnd_diff_str
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
    # 6. SAVE DETAILED PARAMETER TABLES (individual subjects)
    # ------------------------------------------------------------------------
    subject_rows = []
    for subj in subject_ids:
        for cond_idx, cond_label in enumerate(condition_coords):
            pss_m = trace['pss'].sel(subject=subj, condition=cond_label).mean().values
            pss_h = az.hdi(trace['pss'].sel(subject=subj, condition=cond_label).values.flatten(), hdi_prob=0.95)
            slope_m = trace['slope'].sel(subject=subj, condition=cond_label).mean().values
            slope_h = az.hdi(trace['slope'].sel(subject=subj, condition=cond_label).values.flatten(), hdi_prob=0.95)
            jnd_m = trace['threshold'].sel(subject=subj, condition=cond_label).mean().values
            jnd_h = az.hdi(trace['threshold'].sel(subject=subj, condition=cond_label).values.flatten(), hdi_prob=0.95)

            subject_rows.append({
                'subject': subj,
                'condition': condition_labels[cond_idx],
                'pss_mean': pss_m,
                'pss_hdi_lower': pss_h[0],
                'pss_hdi_upper': pss_h[1],
                'slope_mean': slope_m,
                'slope_hdi_lower': slope_h[0],
                'slope_hdi_upper': slope_h[1],
                'jnd_mean': jnd_m,
                'jnd_hdi_lower': jnd_h[0],
                'jnd_hdi_upper': jnd_h[1]
            })
    subject_df = pd.DataFrame(subject_rows)
    subject_csv = os.path.join(output_dir, 'individual_results.csv')
    subject_df.to_csv(subject_csv, index=False)
    print(f"  Saved: {subject_csv}")

    # Population results
    pop_rows = []
    for cond_idx, cond_label in enumerate(condition_coords):
        pss_pop = trace['mu_pss'].sel(condition=cond_label).values.flatten()
        pss_h = az.hdi(pss_pop, hdi_prob=0.95)
        slope_pop = np.exp(trace['mu_slope_log'].sel(condition=cond_label).values.flatten())
        slope_h = az.hdi(slope_pop, hdi_prob=0.95)
        jnd_pop = trace['threshold'].sel(condition=cond_label).values.flatten()
        jnd_h = az.hdi(jnd_pop, hdi_prob=0.95)

        pop_rows.append({
            'condition': condition_labels[cond_idx],
            'pss_mean': np.mean(pss_pop),
            'pss_hdi_lower': pss_h[0],
            'pss_hdi_upper': pss_h[1],
            'slope_mean': np.mean(slope_pop),
            'slope_hdi_lower': slope_h[0],
            'slope_hdi_upper': slope_h[1],
            'jnd_mean': np.mean(jnd_pop),
            'jnd_hdi_lower': jnd_h[0],
            'jnd_hdi_upper': jnd_h[1]
        })
    pop_df = pd.DataFrame(pop_rows)
    pop_csv = os.path.join(output_dir, 'population_results.csv')
    pop_df.to_csv(pop_csv, index=False)
    print(f"  Saved: {pop_csv}")

    # Differences CSV
    diff_df = pd.DataFrame({
        'parameter': ['PSS', 'Slope', 'JND'],
        'mean': [np.mean(pss_diff_vals), np.mean(slope_diff_vals), np.mean(jnd_diff_vals)],
        'hdi_lower': [az.hdi(pss_diff_vals, hdi_prob=0.95)[0],
                      az.hdi(slope_diff_vals, hdi_prob=0.95)[0],
                      az.hdi(jnd_diff_vals, hdi_prob=0.95)[0]],
        'hdi_upper': [az.hdi(pss_diff_vals, hdi_prob=0.95)[1],
                      az.hdi(slope_diff_vals, hdi_prob=0.95)[1],
                      az.hdi(jnd_diff_vals, hdi_prob=0.95)[1]],
        'prob_direction': [pss_pd, slope_pd, jnd_pd]
    })
    diff_csv = os.path.join(output_dir, 'differences.csv')
    diff_df.to_csv(diff_csv, index=False)
    print(f"  Saved: {diff_csv}")

    # ------------------------------------------------------------------------
    # 7. FINAL INTERPRETATION
    # ------------------------------------------------------------------------
    print("\nGenerating final summary text...")
    final_txt = os.path.join(output_dir, 'final_summary.txt')
    with open(final_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TEMPORAL ORDER JUDGMENT ANALYSIS - FINAL SUMMARY (without lapse/guess)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {processed_csv}\n")
        f.write(f"Trace file: {trace_file}\n")
        f.write(f"Number of subjects: {n_subjects}\n\n")

        f.write("POPULATION PARAMETER ESTIMATES (mean [95% HDI]):\n")
        f.write("-" * 60 + "\n")
        for cond_idx, cond_label in enumerate(condition_coords):
            pss_h = az.hdi(trace['mu_pss'].sel(condition=cond_label).values.flatten(), hdi_prob=0.95)
            slope_h = az.hdi(np.exp(trace['mu_slope_log'].sel(condition=cond_label).values.flatten()), hdi_prob=0.95)
            jnd_h = az.hdi(trace['threshold'].sel(condition=cond_label).values.flatten(), hdi_prob=0.95)
            f.write(f"\n{condition_labels[cond_idx]}:\n")
            f.write(f"  PSS:    {mu_pss_mean[cond_idx]:.2f} ms [{pss_h[0]:.2f}, {pss_h[1]:.2f}]\n")
            f.write(f"  Slope:  {mu_slope_mean[cond_idx]:.4f} [{slope_h[0]:.4f}, {slope_h[1]:.4f}]\n")
            f.write(f"  JND:    {jnd_mean[cond_idx]:.2f} ms [{jnd_h[0]:.2f}, {jnd_h[1]:.2f}]\n")

        f.write("\n\nCONDITION DIFFERENCES (Switch - Repeat):\n")
        f.write("-" * 60 + "\n")
        f.write(f"PSS:   {pss_diff_str}   PD = {pss_pd:.1f}%\n")
        f.write(f"Slope: {slope_diff_str} PD = {slope_pd:.1f}%\n")
        f.write(f"JND:   {jnd_diff_str}   PD = {jnd_pd:.1f}%\n")

        f.write("\n\nINTERPRETATION:\n")
        f.write("-" * 60 + "\n")
        pss_diff_hdi = az.hdi(pss_diff_vals, hdi_prob=0.95)
        jnd_diff_hdi = az.hdi(jnd_diff_vals, hdi_prob=0.95)

        if pss_diff_hdi[0] > 0:
            f.write(f"• Switch history shifts PSS to the right by {np.mean(pss_diff_vals):.2f} ms\n")
        elif pss_diff_hdi[1] < 0:
            f.write(f"• Repeat history shifts PSS to the right by {abs(np.mean(pss_diff_vals)):.2f} ms\n")
        else:
            f.write(f"• No credible difference in PSS between conditions\n")

        if jnd_diff_hdi[0] > 0:
            f.write(f"• Switch history shows higher JND (lower sensitivity) by {np.mean(jnd_diff_vals):.2f} ms\n")
        elif jnd_diff_hdi[1] < 0:
            f.write(f"• Repeat history shows higher JND (lower sensitivity) by {abs(np.mean(jnd_diff_vals)):.2f} ms\n")
        else:
            f.write(f"• No credible difference in temporal sensitivity (JND) between conditions\n")

    print(f"  Saved: {final_txt}")

    print("\n" + "=" * 80)
    print("POST‑HOC ANALYSIS COMPLETE (without lapse/guess)")
    print("=" * 80)