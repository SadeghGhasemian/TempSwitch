import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# Load your stats file
# ============================================
os.chdir(r'E:\TemporalSwitch\HDDM_latest\power_analysis_bis_vt\downsampling_results\N35')  # note _vt
stats = pd.read_csv('stats.csv', index_col=0)
print("Stats file loaded successfully")

# ============================================
# Extract parameters for v and t
# ============================================
# Drift rate (v)
v_intercept = stats.loc['v_Intercept', 'mean']
v_hist_S   = stats.loc["v_C(hist, Treatment('R'))[T.S]", 'mean']
v_duration = stats.loc['v_duration', 'mean']
v_interaction = stats.loc["v_C(hist, Treatment('R'))[T.S]:duration", 'mean']

# Non-decision time (t)
t_intercept = stats.loc['t_Intercept', 'mean']
t_hist_S   = stats.loc["t_C(hist, Treatment('R'))[T.S]", 'mean']
t_duration = stats.loc['t_duration', 'mean']
t_interaction = stats.loc["t_C(hist, Treatment('R'))[T.S]:duration", 'mean']

print("\n=== Drift rate (v) parameters ===")
print(f"  v_Intercept: {v_intercept:.4f}")
print(f"  v_hist_S: {v_hist_S:.4f}")
print(f"  v_duration: {v_duration:.4f}")
print(f"  v_interaction: {v_interaction:.4f}")

print("\n=== Non-decision time (t) parameters ===")
print(f"  t_Intercept: {t_intercept:.4f}")
print(f"  t_hist_S: {t_hist_S:.4f}")
print(f"  t_duration: {t_duration:.4f}")
print(f"  t_interaction: {t_interaction:.4f}")

# ============================================
# Duration values (mean-centered)
# ============================================
original_durations = np.array([2, 3, 4, 5, 6, 7, 8])   # seconds? (assuming 2 = 200 ms in your scaling)
duration_mean = original_durations.mean()
durations_centered = original_durations - duration_mean

print(f"\nDuration mean: {duration_mean:.2f} (in your units)")
print("Original → Centered:")
for o, c in zip(original_durations, durations_centered):
    print(f"  {o:.1f} → {c:.3f}")

# ============================================
# Predictions for v and t
# ============================================
# hist = R
v_R = v_intercept + v_duration * durations_centered
t_R = t_intercept + t_duration * durations_centered

# hist = S
v_S = (v_intercept + v_hist_S) + (v_duration + v_interaction) * durations_centered
t_S = (t_intercept + t_hist_S) + (t_duration + t_interaction) * durations_centered

# ============================================
# Display results in tables
# ============================================
results_v = pd.DataFrame({
    'duration_units': original_durations,
    'duration_centered': durations_centered,
    'v_R': v_R,
    'v_S': v_S,
    'difference_v': v_S - v_R
})

results_t = pd.DataFrame({
    'duration_units': original_durations,
    'duration_centered': durations_centered,
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
# NOTE: Multiply original_durations by 100 to get '200','300',... based on your current scaling.
# If your durations are actually seconds, you would multiply by 1000.
tick_positions = original_durations
tick_labels = [f'{int(x*100)}' for x in original_durations]   # adjust factor if needed

# ---- Plot for v ----
plt.figure(figsize=(6, 5))

plt.plot(original_durations, v_R, 'o-', color='k', markersize=7,
         linewidth=1.5, label='Repeat', markeredgecolor='k', markerfacecolor='k')
plt.plot(original_durations, v_S, 'o--', color='0.5', markersize=7,
         linewidth=1.5, label='Switch', markeredgecolor='0.5', markerfacecolor='0.5')

plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.xlabel('Duration (ms)', fontsize=11)
plt.ylabel('Drift rate (v)', fontsize=11)
plt.title('Drift rate (v) by duration and hist condition', fontsize=12, weight='bold')
plt.legend(fontsize=10, frameon=False)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('v_publication.png', dpi=600, bbox_inches='tight')
plt.savefig('v_publication.pdf', bbox_inches='tight')
plt.show()

# ---- Plot for t ----
plt.figure(figsize=(6, 5))

plt.plot(original_durations, t_R, 'o-', color='k', markersize=7,
         linewidth=1.5, label='Repeat', markeredgecolor='k', markerfacecolor='k')
plt.plot(original_durations, t_S, 'o--', color='0.5', markersize=7,
         linewidth=1.5, label='Switch', markeredgecolor='0.5', markerfacecolor='0.5')

plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.xlabel('Duration (ms)', fontsize=11)
plt.ylabel('Non-decision time (t)', fontsize=11)
plt.title('Non-decision time (t) by duration and hist condition', fontsize=12, weight='bold')
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