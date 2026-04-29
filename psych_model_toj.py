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
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("talk")

    # ------------------------------------------------------------------------
    # 0. OUTPUT DIRECTORY SETUP
    # ------------------------------------------------------------------------
    data_path = r"E:\TemporalSwitch\Pymc_psychometric\toj_choice.csv"
    output_dir = r"E:\TemporalSwitch\Pymc_psychometric\output_toj"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MODEL FITTING SCRIPT (without lapse/guess)")
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
    print("SOA values:", np.sort(df['soa'].unique()))
    print("Response distribution:\n", df['response'].value_counts())

    # Create numeric history code: 0 = repeat (R), 1 = switch (S)
    df['hist_code'] = (df['hist'] == 'S').astype(int)

    # Save processed data (fixed filename)
    processed_csv = os.path.join(output_dir, 'processed_data.csv')
    df.to_csv(processed_csv, index=False)
    print(f"\nProcessed data saved to: {processed_csv}")

    # ------------------------------------------------------------------------
    # 2. RAW DATA VISUALIZATION (optional)
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist_val, hist_label, color in zip([0, 1], ['Repeat', 'Switch'], ['blue', 'red']):
        hist_data = df[df['hist_code'] == hist_val]
        prop_data = hist_data.groupby('soa')['response'].agg(['mean', 'count', 'sem']).reset_index()
        prop_data['sem'] = prop_data['sem'].fillna(0)
        axes[0].errorbar(prop_data['soa'], prop_data['mean'],
                         yerr=prop_data['sem'] * 1.96,
                         fmt='o-', color=color, label=hist_label,
                         markersize=8, capsize=3, linewidth=2)

    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('SOA (ms)')
    axes[0].set_ylabel('Proportion "Right" Responses')
    axes[0].set_title('Raw Data: Group Averages')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for subject in df['subject'].unique()[:5]:
        subj_data = df[df['subject'] == subject]
        for hist_val, hist_label, color in zip([0, 1], ['Repeat', 'Switch'], ['blue', 'red']):
            hist_subj = subj_data[subj_data['hist_code'] == hist_val]
            if len(hist_subj) > 0:
                prop_subj = hist_subj.groupby('soa')['response'].mean().reset_index()
                axes[1].plot(prop_subj['soa'], prop_subj['response'],
                             color=color, alpha=0.3, linewidth=1)

    for hist_val, hist_label, color in zip([0, 1], ['Repeat', 'Switch'], ['blue', 'red']):
        hist_data = df[df['hist_code'] == hist_val]
        prop_data = hist_data.groupby('soa')['response'].mean().reset_index()
        axes[1].plot(prop_data['soa'], prop_data['response'],
                     color=color, linewidth=3, label=f'{hist_label} (avg)')

    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('SOA (ms)')
    axes[1].set_ylabel('Proportion "Right" Responses')
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
    # Create coordinate values (1D arrays/lists only)
    subject_ids = np.sort(df['subject'].unique())
    condition_names = ['repeat', 'switch']  # 0 and 1
    coords = {
        'subject': subject_ids,
        'condition': condition_names,
    }

    # Indices for subject and condition
    subjects = pd.Categorical(df['subject'], categories=subject_ids).codes
    conditions = df['hist_code'].values
    soas = df['soa'].values.astype(float)
    responses = df['response'].values

    n_subjects = len(subject_ids)
    n_conditions = 2

    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    print(f"Number of subjects: {n_subjects}")
    print(f"Number of conditions: {n_conditions} (0=Repeat, 1=Switch)")
    print(f"Total observations: {len(responses)}")
    print(f"SOA range: [{soas.min()}, {soas.max()}] ms")

    # ------------------------------------------------------------------------
    # 4. BUILD MODEL WITHOUT LAPSE/GUESS
    # ------------------------------------------------------------------------
    with pm.Model(coords=coords) as toj_model:
        # Population-level hyperpriors
        mu_pss = pm.Normal('mu_pss', mu=0, sigma=50, dims='condition')
        sigma_pss = pm.HalfNormal('sigma_pss', sigma=25, dims='condition')

        mu_slope_log = pm.Normal('mu_slope_log', mu=np.log(0.02), sigma=1, dims='condition')
        sigma_slope_log = pm.HalfNormal('sigma_slope_log', sigma=0.5, dims='condition')

        # Subject-level random effects for PSS and slope
        pss_offset = pm.Normal('pss_offset', mu=0, sigma=1, dims=('subject', 'condition'))
        pss = pm.Deterministic('pss', mu_pss + pss_offset * sigma_pss, dims=('subject', 'condition'))

        slope_log_offset = pm.Normal('slope_log_offset', mu=0, sigma=1, dims=('subject', 'condition'))
        slope_log = mu_slope_log + slope_log_offset * sigma_slope_log
        slope = pm.Deterministic('slope', pt.exp(slope_log), dims=('subject', 'condition'))

        # Derived quantity: JND
        threshold = pm.Deterministic('threshold', 1.35 / slope, dims=('subject', 'condition'))

        # Psychometric function (probit, no lapse/guess)
        pss_obs = pss[subjects, conditions]
        slope_obs = slope[subjects, conditions]
        prob_right = pm.math.invprobit((soas - pss_obs) * slope_obs)

        # Likelihood
        likelihood = pm.Bernoulli('likelihood', p=prob_right, observed=responses)

    print("\n" + "=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    print(toj_model)

    model_txt = os.path.join(output_dir, 'model_specification.txt')
    with open(model_txt, 'w') as f:
        f.write(str(toj_model))
    print(f"Model specification saved to: {model_txt}")

    # ------------------------------------------------------------------------
    # 5. FIT MODEL AND SAMPLE POSTERIOR PREDICTIVE
    # ------------------------------------------------------------------------
    with toj_model:
        print("\nStarting sampling...")
        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            cores=4,
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
    rhat = az.rhat(idata, var_names=['mu_pss', 'mu_slope_log'])
    ess = az.ess(idata, var_names=['mu_pss', 'mu_slope_log'])

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