import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import warnings
import os
import json

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # ------------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------------
    base_dir = r"E:\TemporalSwitch\Pymc_reaction_time"
    data_path = os.path.join(base_dir, "toj_rt.csv")            # your TOJ RT data file
    output_dir = os.path.join(base_dir, "output_rt_toj_centered_28ms")   # new output folder
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MODEL FITTING – REACTION TIME (TEMPORAL ORDER JUDGMENT) – MEAN‑CENTERED PREDICTORS")
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
    print("Block levels:", df['block'].unique())
    print("RSI values (s):", df['rsi'].unique())
    print("SOA values (ms):", np.sort(df['soa'].unique()))

    # Encode history as 0 (repeat) and 1 (switch)
    df['hist_code'] = (df['hist'] == 'S').astype(int)
    condition_labels = ['Repeat', 'Switch']

    # RT in milliseconds
    df['rt_ms'] = df['rt'] * 1000

    # Convert SOA to 28 ms units (divide by 28)
    df['soa_28ms'] = df['soa'] / 28.0

    # Convert RSI to 100 ms units (multiply by 10)
    df['rsi_100ms'] = df['rsi'] * 10.0

    # Block remains integer (1‑4)

    # Compute means for centering
    soa_mean = df['soa_28ms'].mean()
    rsi_mean = df['rsi_100ms'].mean()
    block_mean = df['block'].mean()

    # Mean‑center predictors
    df['soa_c'] = df['soa_28ms'] - soa_mean
    df['rsi_c'] = df['rsi_100ms'] - rsi_mean
    df['block_c'] = df['block'] - block_mean

    # Save centering parameters for later use
    center_params = {
        'soa_mean': soa_mean,
        'rsi_mean': rsi_mean,
        'block_mean': block_mean,
        'soa_unit': '28ms',
        'rsi_unit': '100ms',
        'block_unit': 'blocks'
    }
    with open(os.path.join(output_dir, 'centering_params.json'), 'w') as f:
        json.dump(center_params, f)

    # Save processed data
    processed_csv = os.path.join(output_dir, 'rt_processed_data.csv')
    df.to_csv(processed_csv, index=False)
    print(f"\nProcessed RT data (centered) saved to: {processed_csv}")

    # ------------------------------------------------------------------------
    # 2. PREPARE DATA FOR PYMC
    # ------------------------------------------------------------------------
    subject_ids = np.sort(df['subject'].unique())
    n_subjects = len(subject_ids)

    subj_idx = pd.Categorical(df['subject'], categories=subject_ids).codes
    hist = df['hist_code'].values.astype(float)
    soa_c = df['soa_c'].values.astype(float)
    block_c = df['block_c'].values.astype(float)
    rsi_c = df['rsi_c'].values.astype(float)
    rt_ms = df['rt_ms'].values

    # Quadratic term (centered SOA)
    soa_c2 = soa_c ** 2

    # Interaction terms
    hist_soa = hist * soa_c
    hist_soa2 = hist * soa_c2
    hist_block = hist * block_c
    hist_rsi = hist * rsi_c

    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    print(f"Number of subjects: {n_subjects}")
    print(f"Total observations: {len(rt_ms)}")
    print(f"RT range (ms): [{rt_ms.min():.0f}, {rt_ms.max():.0f}]")
    print(f"SOA (centered) range: [{soa_c.min():.1f}, {soa_c.max():.1f}] (28 ms units)")
    print(f"Block (centered) range: [{block_c.min():.1f}, {block_c.max():.1f}]")
    print(f"RSI (centered) range: [{rsi_c.min():.1f}, {rsi_c.max():.1f}] (100 ms units)")

    # Coordinates for ArviZ
    coords = {
        'subject': subject_ids,
        'condition': condition_labels,
        'predictor': ['intercept', 'hist', 'soa', 'soa2', 'block', 'rsi',
                      'hist_soa', 'hist_soa2', 'hist_block', 'hist_rsi']
    }

    # ------------------------------------------------------------------------
    # 3. BUILD HIERARCHICAL MODEL
    # ------------------------------------------------------------------------
    with pm.Model(coords=coords) as rt_model:
        # Population-level fixed effects
        # Intercept: mean RT at mean predictors (centered = 0)
        mu_intercept = pm.Normal('mu_intercept', mu=500, sigma=200)

        # History effect (additive ms)
        mu_hist = pm.Normal('mu_hist', mu=0, sigma=100)

        # SOA linear effect (per 28 ms)
        mu_soa = pm.Normal('mu_soa', mu=0, sigma=50)

        # SOA quadratic effect (per (28 ms)²)
        mu_soa2 = pm.Normal('mu_soa2', mu=0, sigma=30)

        # Block effect (per block)
        mu_block = pm.Normal('mu_block', mu=0, sigma=30)

        # RSI effect (per 100 ms)
        mu_rsi = pm.Normal('mu_rsi', mu=0, sigma=30)

        # Interactions
        mu_hist_soa = pm.Normal('mu_hist_soa', mu=0, sigma=50)
        mu_hist_soa2 = pm.Normal('mu_hist_soa2', mu=0, sigma=30)
        mu_hist_block = pm.Normal('mu_hist_block', mu=0, sigma=30)
        mu_hist_rsi = pm.Normal('mu_hist_rsi', mu=0, sigma=30)

        # Random effects (subject-specific)
        sigma_subj_intercept = pm.HalfNormal('sigma_subj_intercept', sigma=100)
        z_subj_intercept = pm.Normal('z_subj_intercept', mu=0, sigma=1, dims='subject')
        subj_intercept = pm.Deterministic('subj_intercept', mu_intercept + z_subj_intercept * sigma_subj_intercept, dims='subject')

        sigma_subj_hist = pm.HalfNormal('sigma_subj_hist', sigma=50)
        z_subj_hist = pm.Normal('z_subj_hist', mu=0, sigma=1, dims='subject')
        subj_hist = pm.Deterministic('subj_hist', mu_hist + z_subj_hist * sigma_subj_hist, dims='subject')

        # Linear predictor (RT in ms)
        mu_rt = (subj_intercept[subj_idx] +
                 subj_hist[subj_idx] * hist +
                 mu_soa * soa_c +
                 mu_soa2 * soa_c2 +
                 mu_block * block_c +
                 mu_rsi * rsi_c +
                 mu_hist_soa * hist_soa +
                 mu_hist_soa2 * hist_soa2 +
                 mu_hist_block * hist_block +
                 mu_hist_rsi * hist_rsi)

        # Likelihood (Normal on ms)
        sigma = pm.HalfNormal('sigma', sigma=100)
        likelihood = pm.Normal('likelihood', mu=mu_rt, sigma=sigma, observed=rt_ms)

    print("\n" + "=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    print(rt_model)

    model_txt = os.path.join(output_dir, 'rt_model_specification.txt')
    with open(model_txt, 'w') as f:
        f.write(str(rt_model))
    print(f"Model specification saved to: {model_txt}")

    # ------------------------------------------------------------------------
    # 4. FIT MODEL
    # ------------------------------------------------------------------------
    with rt_model:
        print("\nStarting sampling (adapt_delta=0.9, tune=1500, draws=1500, chains=4)...")
        idata = pm.sample(
            draws=1500,
            tune=1500,
            chains=4,
            cores=4,
            adapt_delta=0.9,
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True},
            progressbar=True
        )
        # Posterior predictive (for later use)
        pm.sample_posterior_predictive(idata, random_seed=42, extend_inferencedata=True)

    # Save trace
    trace_file = os.path.join(output_dir, 'rt_trace.nc')
    idata.to_netcdf(trace_file)
    print(f"\nTrace saved to: {trace_file}")

    # ------------------------------------------------------------------------
    # 5. QUICK CONVERGENCE CHECK
    # ------------------------------------------------------------------------
    rhat = az.rhat(idata, var_names=['mu_intercept', 'mu_hist', 'mu_soa', 'mu_soa2',
                                      'mu_block', 'mu_rsi', 'mu_hist_soa', 'mu_hist_soa2',
                                      'mu_hist_block', 'mu_hist_rsi', 'sigma'])
    ess = az.ess(idata, var_names=['mu_intercept', 'mu_hist', 'mu_soa', 'mu_soa2',
                                    'mu_block', 'mu_rsi', 'mu_hist_soa', 'mu_hist_soa2',
                                    'mu_hist_block', 'mu_hist_rsi', 'sigma'])

    print("\n" + "=" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 70)
    print("R-hat (should be < 1.01):\n", rhat)
    print("\nEffective sample size:\n", ess)

    diag_file = os.path.join(output_dir, 'rt_convergence.txt')
    with open(diag_file, 'w') as f:
        f.write("R-hat values:\n")
        f.write(str(rhat))
        f.write("\n\nEffective sample size:\n")
        f.write(str(ess))
    print(f"Convergence diagnostics saved to: {diag_file}")

    print("\n" + "=" * 80)
    print("MODEL FITTING COMPLETE")
    print("=" * 80)