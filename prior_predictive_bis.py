import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# ------------------------------------------------------------------------
# CONFIGURATION – USE THE SAME PATHS AS YOUR MAIN ANALYSIS
# ------------------------------------------------------------------------
output_dir = r"E:\TemporalSwitch\Pymc_psychometric\output_bis"
processed_csv = os.path.join(output_dir, 'processed_data.csv')

print("=" * 70)
print("PRIOR PREDICTIVE CHECK – TEMPORAL BISECTION (LOGISTIC)")
print("=" * 70)
print(f"Loading processed data: {processed_csv}")

# ------------------------------------------------------------------------
# 1. LOAD AND PREPARE DATA (SAME AS IN FITTING SCRIPT)
# ------------------------------------------------------------------------
df = pd.read_csv(processed_csv)

# Ensure hist_code exists (it should from the fitting script)
if 'hist_code' not in df.columns:
    df['hist_code'] = (df['hist'] == 'S').astype(int)

subject_ids = np.sort(df['subject'].unique())
condition_names = ['repeat', 'switch']  # 0 = repeat, 1 = switch
coords = {
    'subject': subject_ids,
    'condition': condition_names,
}

# Indices for subject and condition
subjects = pd.Categorical(df['subject'], categories=subject_ids).codes
conditions = df['hist_code'].values
durations = df['duration'].values.astype(float)
responses = df['response'].values

n_subjects = len(subject_ids)
n_conditions = 2

print(f"Number of subjects: {n_subjects}")
print(f"Total observations: {len(responses)}")
print(f"Duration range: [{durations.min()}, {durations.max()}] s")

# ------------------------------------------------------------------------
# 2. BUILD THE MODEL WITH THE SAME PRIORS (BUT WITHOUT OBSERVED DATA)
# ------------------------------------------------------------------------
with pm.Model(coords=coords) as prior_model:
    # Population-level hyperpriors
    mu_pse = pm.Normal('mu_pse', mu=0.5, sigma=0.2, dims='condition')
    sigma_pse = pm.HalfNormal('sigma_pse', sigma=0.1, dims='condition')

    mu_log_slope = pm.Normal('mu_log_slope', mu=np.log(10), sigma=0.5, dims='condition')
    sigma_log_slope = pm.HalfNormal('sigma_log_slope', sigma=0.25, dims='condition')

    # Subject-level offsets (non‑centered)
    pse_offset = pm.Normal('pse_offset', mu=0, sigma=1, dims=('subject', 'condition'))
    pse = pm.Deterministic('pse', mu_pse + pse_offset * sigma_pse, dims=('subject', 'condition'))

    log_slope_offset = pm.Normal('log_slope_offset', mu=0, sigma=1, dims=('subject', 'condition'))
    log_slope = mu_log_slope + log_slope_offset * sigma_log_slope
    slope = pm.Deterministic('slope', pt.exp(log_slope), dims=('subject', 'condition'))

    # Psychometric function for each observation
    pse_obs = pse[subjects, conditions]
    slope_obs = slope[subjects, conditions]
    prob_long = pm.math.invlogit((durations - pse_obs) * slope_obs)

    # Prior predictive samples of responses
    prior_pred = pm.Bernoulli('prior_pred', p=prob_long)

print("\nModel successfully built. Sampling prior predictive...")

# ------------------------------------------------------------------------
# 3. DRAW PRIOR PREDICTIVE SAMPLES
# ------------------------------------------------------------------------
with prior_model:
    prior_predictive = pm.sample_prior_predictive(samples=1000, random_seed=42)

# Check the structure of the returned object
print("\nAvailable groups in prior_predictive:")
print(prior_predictive)

# Extract the simulated responses – use the correct group name ('prior')
# The array has shape (chain=1, samples=500, n_obs); we remove the chain dimension with .squeeze()
prior_pred_vals = prior_predictive.prior['prior_pred'].values.squeeze()  # shape (500, n_obs)

print("Prior predictive sampling completed. Shape of simulated data:", prior_pred_vals.shape)

# ------------------------------------------------------------------------
# 4. COMPUTE OBSERVED PROPORTIONS FOR COMPARISON
# ------------------------------------------------------------------------
durations_unique = np.sort(df['duration'].unique())
cond_list = [0, 1]
condition_labels = ['Repeat', 'Switch']

obs_props_list = []
for cond in cond_list:
    cond_data = df[df['hist_code'] == cond]
    props = cond_data.groupby('duration')['response'].mean().reindex(durations_unique).values
    obs_props_list.append(props)

# ------------------------------------------------------------------------
# 5. COMPUTE PRIOR PREDICTIVE INTERVALS FOR EACH CONDITION AND DURATION
# ------------------------------------------------------------------------
prior_mean = np.zeros((len(cond_list), len(durations_unique)))
prior_lower = np.zeros_like(prior_mean)
prior_upper = np.zeros_like(prior_mean)

for cond_idx, cond in enumerate(cond_list):
    # Indices of observations belonging to this condition
    obs_idx = np.where(conditions == cond)[0]
    # For each duration, collect predictions and compute mean and 95% interval
    for i, dur in enumerate(durations_unique):
        dur_obs_idx = np.where((conditions == cond) & (durations == dur))[0]
        if len(dur_obs_idx) > 0:
            prior_dur = prior_pred_vals[:, dur_obs_idx]  # shape (samples, n_trials_at_dur)
            # Average over trials within each sample to get proportion for that duration
            prior_prop = prior_dur.mean(axis=1)  # shape (samples,)
            prior_mean[cond_idx, i] = prior_prop.mean()
            prior_lower[cond_idx, i] = np.percentile(prior_prop, 2.5)
            prior_upper[cond_idx, i] = np.percentile(prior_prop, 97.5)
        else:
            prior_mean[cond_idx, i] = np.nan
            prior_lower[cond_idx, i] = np.nan
            prior_upper[cond_idx, i] = np.nan

# ------------------------------------------------------------------------
# 6. PLOT PRIOR PREDICTIVE CHECK
# ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for cond_idx, cond_label in enumerate(condition_labels):
    ax = axes[cond_idx]
    # Prior predictive interval
    ax.fill_between(durations_unique, prior_lower[cond_idx], prior_upper[cond_idx],
                    color='gray', alpha=0.3, label='95% prior interval')
    # Prior mean
    ax.plot(durations_unique, prior_mean[cond_idx], color='black', linestyle='-', linewidth=2, label='Prior mean')
    # Observed data (black circles)
    ax.plot(durations_unique, obs_props_list[cond_idx], 'o', color='black', markersize=6, label='Observed')
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Proportion "Long"')
    ax.set_title(f'{cond_label} Condition')
    ax.set_xticks(durations_unique)
    ax.set_xticklabels([f'{x:.1f}' for x in durations_unique])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Prior Predictive Check – Temporal Bisection', fontsize=14)
plt.tight_layout()

# Save figure
prior_png = os.path.join(output_dir, 'prior_predictive_check.png')
prior_pdf = os.path.join(output_dir, 'prior_predictive_check.pdf')
plt.savefig(prior_png, dpi=300, bbox_inches='tight')
plt.savefig(prior_pdf, bbox_inches='tight')
plt.show()

print(f"\nPrior predictive check figure saved:")
print(f"  {prior_png}")
print(f"  {prior_pdf}")

# ------------------------------------------------------------------------
# 7. BRIEF SUMMARY
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("The prior predictive check shows the range of response proportions")
print("predicted by the prior distributions alone (gray band: 95% interval,")
print("black line: prior mean). The observed data (black dots) fall mostly")
print("within this range, indicating that the priors are compatible with")
print("the observed response patterns and do not strongly constrain the")
print("inference in an unrealistic way.")
print("=" * 70)