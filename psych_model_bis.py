import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytensor.tensor as pt
import warnings
import os

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # ------------------------------------------------------------------------
    # 0. OUTPUT DIRECTORY SETUP
    # ------------------------------------------------------------------------
    data_path = r"E:\TemporalSwitch\Pymc_psychometric\bis_choice.csv"
    output_dir = r"E:\TemporalSwitch\Pymc_psychometric\output_bis"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MODEL FITTING SCRIPT – TEMPORAL BISECTION (LOGISTIC)")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------------
    # 1. LOAD AND PREPROCESS DATA
    # ------------------------------------------------------------------------
    df = pd.read_csv(data_path)

    print("\nData shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Subjects:", df['subject'].nunique())
    print("History levels:", df['hist'].unique())
    print("Duration values:", np.sort(df['duration'].unique()))
    print("Response distribution (1 = long, 0 = short):\n", df['response'].value_counts())

    df['hist_code'] = (df['hist'] == 'S').astype(int)
    processed_csv = os.path.join(output_dir, 'processed_data.csv')
    df.to_csv(processed_csv, index=False)
    print(f"\nProcessed data saved to: {processed_csv}")

    # ------------------------------------------------------------------------
    # 2. RAW DATA VISUALIZATION (optional)
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist_val, hist_label, color in zip([0, 1], ['Repeat', 'Switch'], ['blue', 'red']):
        hist_data = df[df['hist_code'] == hist_val]
        prop_data = hist_data.groupby('duration')['response'].agg(['mean', 'count', 'sem']).reset_index()
        prop_data['sem'] = prop_data['sem'].fillna(0)
        axes[0].errorbar(prop_data['duration'], prop_data['mean'],
                         yerr=prop_data['sem'] * 1.96,
                         fmt='o-', color=color, label=hist_label,
                         markersize=8, capsize=3, linewidth=2)

    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    axes[0].set_xlabel('Duration (s)')
    axes[0].set_ylabel('Proportion "Long" Responses')
    axes[0].set_title('Raw Data: Group Averages')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for subject in df['subject'].unique()[:5]:
        subj_data = df[df['subject'] == subject]
        for hist_val, hist_label, color in zip([0, 1], ['Repeat', 'Switch'], ['blue', 'red']):
            hist_subj = subj_data[subj_data['hist_code'] == hist_val]
            if len(hist_subj) > 0:
                prop_subj = hist_subj.groupby('duration')['response'].mean().reset_index()
                axes[1].plot(prop_subj['duration'], prop_subj['response'],
                             color=color, alpha=0.3, linewidth=1)

    for hist_val, hist_label, color in zip([0, 1], ['Repeat', 'Switch'], ['blue', 'red']):
        hist_data = df[df['hist_code'] == hist_val]
        prop_data = hist_data.groupby('duration')['response'].mean().reset_index()
        axes[1].plot(prop_data['duration'], prop_data['response'],
                     color=color, linewidth=3, label=f'{hist_label} (avg)')

    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Duration (s)')
    axes[1].set_ylabel('Proportion "Long" Responses')
    axes[1].set_title('Individual Subject Data (first 5 subjects)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    raw_png = os.path.join(output_dir, 'raw_data.png')
    raw_pdf = os.path.join(output_dir, 'raw_data.pdf')
    plt.savefig(raw_png, dpi=300, bbox_inches='tight')
    plt.savefig(raw_pdf, bbox_inches='tight')
    plt.show()
    print(f"Raw data plots saved: {raw_png}")

    # ------------------------------------------------------------------------
    # 3. PREPARE DATA FOR PYMC
    # ------------------------------------------------------------------------
    subject_ids = np.sort(df['subject'].unique())
    condition_names = ['repeat', 'switch']
    coords = {
        'subject': subject_ids,
        'condition': condition_names,
    }

    subjects = pd.Categorical(df['subject'], categories=subject_ids).codes
    conditions = df['hist_code'].values
    durations = df['duration'].values.astype(float)
    responses = df['response'].values

    n_subjects = len(subject_ids)
    n_conditions = 2

    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    print(f"Number of subjects: {n_subjects}")
    print(f"Number of conditions: {n_conditions} (0=Repeat, 1=Switch)")
    print(f"Total observations: {len(responses)}")
    print(f"Duration range: [{durations.min()}, {durations.max()}] s")

    # ------------------------------------------------------------------------
    # 4. BUILD LOGISTIC HIERARCHICAL MODEL WITH TIGHTER PRIORS
    # ------------------------------------------------------------------------
    with pm.Model(coords=coords) as bis_model:
        # Population-level hyperpriors for PSE (location)
        mu_pse = pm.Normal('mu_pse', mu=0.5, sigma=0.2, dims='condition')
        sigma_pse = pm.HalfNormal('sigma_pse', sigma=0.1, dims='condition')

        # Population-level hyperpriors for log-slope (sensitivity)
        # Tighter prior: sigma=0.5 instead of 1
        mu_log_slope = pm.Normal('mu_log_slope', mu=np.log(10), sigma=0.5, dims='condition')
        # Tighter prior on variance
        sigma_log_slope = pm.HalfNormal('sigma_log_slope', sigma=0.25, dims='condition')

        # Subject-level random effects (non‑centered)
        pse_offset = pm.Normal('pse_offset', mu=0, sigma=1, dims=('subject', 'condition'))
        pse = pm.Deterministic('pse', mu_pse + pse_offset * sigma_pse, dims=('subject', 'condition'))

        log_slope_offset = pm.Normal('log_slope_offset', mu=0, sigma=1, dims=('subject', 'condition'))
        log_slope = mu_log_slope + log_slope_offset * sigma_log_slope
        slope = pm.Deterministic('slope', pt.exp(log_slope), dims=('subject', 'condition'))

        # Psychometric function (logistic)
        pse_obs = pse[subjects, conditions]
        slope_obs = slope[subjects, conditions]
        prob_long = pm.math.invlogit((durations - pse_obs) * slope_obs)

        # Likelihood
        likelihood = pm.Bernoulli('likelihood', p=prob_long, observed=responses)

    print("\n" + "=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    print(bis_model)

    model_txt = os.path.join(output_dir, 'model_specification.txt')
    with open(model_txt, 'w') as f:
        f.write(str(bis_model))
    print(f"Model specification saved to: {model_txt}")

    # ------------------------------------------------------------------------
    # 5. FIT MODEL WITH HIGHER ADAPT_DELTA AND LONGER TUNE
    # ------------------------------------------------------------------------
    with bis_model:
        print("\nStarting sampling with adapt_delta=0.90, tune=2000...")
        idata = pm.sample(
            draws=3000,
            tune=3000,                # longer warmup
            chains=4,
            cores=4,
            adapt_delta=0.95,          # much smaller steps
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True},
            progressbar=True
        )

        print("\nSampling posterior predictive...")
        pm.sample_posterior_predictive(
            idata,
            random_seed=42,
            extend_inferencedata=True,
            progressbar=True
        )

    # ------------------------------------------------------------------------
    # 6. SAVE THE COMPLETE INFERENCEDATA
    # ------------------------------------------------------------------------
    trace_file = os.path.join(output_dir, 'trace.nc')
    idata.to_netcdf(trace_file)
    print(f"\nFull trace (with posterior predictive) saved to: {trace_file}")

    # ------------------------------------------------------------------------
    # 7. BASIC CONVERGENCE DIAGNOSTICS
    # ------------------------------------------------------------------------
    rhat = az.rhat(idata, var_names=['mu_pse', 'mu_log_slope'])
    ess = az.ess(idata, var_names=['mu_pse', 'mu_log_slope'])

    print("\n" + "=" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 70)
    print("R-hat (should be < 1.01):\n", rhat)
    print("\nEffective sample size:\n", ess)

    diag_file = os.path.join(output_dir, 'convergence.txt')
    with open(diag_file, 'w') as f:
        f.write("R-hat values:\n")
        f.write(str(rhat))
        f.write("\n\nEffective sample size:\n")
        f.write(str(ess))
    print(f"Convergence diagnostics saved to: {diag_file}")

    print("\n" + "=" * 70)
    print("MODEL FITTING COMPLETE")
    print("=" * 70)