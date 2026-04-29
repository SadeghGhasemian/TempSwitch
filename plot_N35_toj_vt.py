import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np

# Original values
values = np.array([0.1, 0.072, 0.044, 0.016])

# Scale each element by (100/28)
scaled = values * (100 / 28)

# Compute the mean of the scaled array
mean_val = np.mean(scaled)

# Subtract the mean (center the data)
centered = scaled - mean_val

print(centered)

# ============================================
# Load your stats file
# ============================================
os.chdir(r'E:\TemporalSwitch\HDDM_latest\power_analysis_toj_vt\downsampling_results\N35')  # note _vt
stats = pd.read_csv('stats.csv', index_col=0)
print("Stats file loaded successfully")

# ============================================
# Extract parameters for v and t
# ============================================
# Drift rate (v)
v_intercept = stats.loc['v_Intercept', 'mean']
v_hist_S   = stats.loc["v_C(hist, Treatment('R'))[T.S]", 'mean']
v_soa = stats.loc['v_soa', 'mean']
v_interaction = stats.loc["v_C(hist, Treatment('R'))[T.S]:soa", 'mean']

# Non-decision time (t)
t_intercept = stats.loc['t_Intercept', 'mean']
t_hist_S   = stats.loc["t_C(hist, Treatment('R'))[T.S]", 'mean']
t_soa = stats.loc['t_soa', 'mean']
t_interaction = stats.loc["t_C(hist, Treatment('R'))[T.S]:soa", 'mean']

print("\n=== Drift rate (v) parameters ===")
print(f"  v_Intercept: {v_intercept:.4f}")
print(f"  v_hist_S: {v_hist_S:.4f}")
print(f"  v_soa: {v_soa:.4f}")
print(f"  v_interaction: {v_interaction:.4f}")

print("\n=== Non-decision time (t) parameters ===")
print(f"  t_Intercept: {t_intercept:.4f}")
print(f"  t_hist_S: {t_hist_S:.4f}")
print(f"  t_soa: {t_soa:.4f}")
print(f"  t_interaction: {t_interaction:.4f}")

# ============================================
# soa values (mean-centered)
# ============================================
original_soas = np.array([0.016, 0.044, 0.072, 0.1])*100/28   # seconds? 
soa_mean = original_soas.mean()
soas_centered = (original_soas - soa_mean)
print(soas_centered)

print(f"\nsoa mean: {soa_mean:.2f} (in your units)")
print("Original → Centered:")
for o, c in zip(original_soas, soas_centered):
    print(f"  {o:.1f} → {c:.3f}")

# ============================================
# Predictions for v and t
# ============================================
# hist = R
v_R = v_intercept + v_soa/10 * soas_centered*10
t_R = t_intercept + t_soa/10 * soas_centered*10

# hist = S
v_S = (v_intercept + v_hist_S) + (v_soa/10 + v_interaction/10) * soas_centered*10
t_S = (t_intercept + t_hist_S) + (t_soa/10 + t_interaction/10) * soas_centered*10

# ============================================
# Display results in tables
# ============================================
results_v = pd.DataFrame({
    'soa_units': original_soas,
    'soa_centered': soas_centered,
    'v_R': v_R,
    'v_S': v_S,
    'difference_v': v_S - v_R
})

results_t = pd.DataFrame({
    'soa_units': original_soas,
    'soa_centered': soas_centered,
    't_R': t_R,
    't_S': t_S,
    'difference_t': t_S - t_R
})

print("\n" + "="*60)
print("PREDICTED v VALUES")
print("="*60)
print(results_v.round(4).to_string(index=False))

print("\n" + "="*60)
print("PREDICTED t VALUES")
print("="*60)
print(results_t.round(4).to_string(index=False))

# ============================================
# Publication-ready grayscale plots (v and t)
# ============================================
# Set up the x‑axis: ticks at the data points, labelled as milliseconds
# NOTE: Multiply original_soas by 100 to get '200','300',... based on your current scaling.
# If your soas are actually seconds, you would multiply by 1000.
tick_positions = original_soas

tick_labels = ['16', '44', '72', '100']

# ---- Plot for v ----
plt.figure(figsize=(6, 5))

plt.plot(original_soas, v_R, 'o-', color='k', markersize=7,
         linewidth=1.5, label='Repeat', markeredgecolor='k', markerfacecolor='k')
plt.plot(original_soas, v_S, 'o--', color='0.5', markersize=7,
         linewidth=1.5, label='Switch', markeredgecolor='0.5', markerfacecolor='0.5')

plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.xlabel('soa (ms)', fontsize=11)
plt.ylabel('Drift rate (v)', fontsize=11)
plt.title('Drift rate (v) by soa and hist condition', fontsize=12, weight='bold')
plt.legend(fontsize=10, frameon=False)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('v_publication.png', dpi=600, bbox_inches='tight')
plt.savefig('v_publication.pdf', bbox_inches='tight')
plt.show()

# ---- Plot for t ----
plt.figure(figsize=(6, 5))

plt.plot(original_soas, t_R*1000, 'o-', color='k', markersize=7,
         linewidth=1.5, label='Repeat', markeredgecolor='k', markerfacecolor='k')
plt.plot(original_soas, t_S*1000, 'o--', color='0.5', markersize=7,
         linewidth=1.5, label='Switch', markeredgecolor='0.5', markerfacecolor='0.5')

plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.xlabel('soa (ms)', fontsize=11)
plt.ylabel('Non-decision time (t)', fontsize=11)

plt.title('Non-decision time (t) by soa and hist condition', fontsize=12, weight='bold')
plt.legend(fontsize=10, frameon=False)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('t_publication.png', dpi=600, bbox_inches='tight')
plt.savefig('t_publication.pdf', bbox_inches='tight')
plt.show()

# ============================================
# Save predictions to CSV
# ============================================
results_v.to_csv('v_predictions.csv', index=False)
results_t.to_csv('t_predictions.csv', index=False)
print("\n✅ Saved predictions and publication figures for v and t.")