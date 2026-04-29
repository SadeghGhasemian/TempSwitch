import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# Load your stats file
# ============================================
os.chdir(r'E:\TemporalSwitch\HDDM_latest\power_analysis_bis_z\downsampling_results\N35')
stats = pd.read_csv('stats.csv', index_col=0)
print("Stats file loaded successfully")

# ============================================
# Extract parameters directly from stats
# ============================================
z_intercept = stats.loc['z_Intercept', 'mean']
z_hist_S = stats.loc["z_C(hist, Treatment('R'))[T.S]", 'mean']
z_duration = stats.loc['z_duration', 'mean']
z_interaction = stats.loc["z_C(hist, Treatment('R'))[T.S]:duration", 'mean']

print(f"\nParameters:")
print(f"  z_Intercept: {z_intercept:.4f}")
print(f"  z_hist_S: {z_hist_S:.4f}")
print(f"  z_duration: {z_duration:.4f}")
print(f"  z_interaction: {z_interaction:.4f}")

# ============================================
# Duration values (mean-centered only)
# ============================================
original_durations = np.array([2, 3, 4, 5, 6, 7, 8])
duration_mean = original_durations.mean()
durations_centered = original_durations - duration_mean

print(f"\nDuration mean: {duration_mean:.2f}s")
print("Original → Centered:")
for o, c in zip(original_durations, durations_centered):
    print(f"  {o:.1f}s → {c:.3f}")

# ============================================
# Simple linear equations (NO transformations)
# ============================================
# hist = R: z = intercept + duration * d
z_R = z_intercept + z_duration * durations_centered

# hist = S: z = intercept + hist_effect + (duration + interaction) * d
z_S = (z_intercept + z_hist_S) + (z_duration + z_interaction) * durations_centered

# ============================================
# Display results
# ============================================
results = pd.DataFrame({
    'duration_sec': original_durations,
    'duration_centered': durations_centered,
    'z_R': z_R,
    'z_S': z_S,
    'difference': z_S - z_R
})

print("\n" + "="*60)
print("PREDICTED z VALUES")
print("="*60)
print(results.round(4).to_string(index=False))

# ============================================
# Publication-ready grayscale plot with ms x-axis
# ============================================
plt.figure(figsize=(6, 5))

# Plot hist=R with black solid line and circles
plt.plot(original_durations, z_R, 'o-', color='k', markersize=7,
         linewidth=1.5, label='Repeat', markeredgecolor='k', markerfacecolor='k')

# Plot hist=S with gray dashed line and squares
plt.plot(original_durations, z_S, 'o--', color='0.5', markersize=7,
         linewidth=1.5, label='Switch', markeredgecolor='0.5', markerfacecolor='0.5')

# Customize x-axis: set ticks at the exact data points (in seconds)
# but label them as milliseconds (multiply by 100 and format as int)
tick_positions = original_durations  # in seconds
tick_labels = [f'{int(ms)}' for ms in original_durations * 100]  # e.g., '200', '300', ...
plt.xticks(ticks=tick_positions, labels=tick_labels)

plt.xlabel('Duration (ms)', fontsize=11)
plt.ylabel('Starting point (z)', fontsize=11)
plt.title('Starting point (z) by duration and hist condition', fontsize=12, weight='bold')
plt.legend(fontsize=10, frameon=False)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save high-resolution PNG and PDF
plt.savefig('z_publication.png', dpi=600, bbox_inches='tight')
plt.savefig('z_publication.pdf', bbox_inches='tight')
plt.show()

# Save to CSV
results.to_csv('z_predictions_simple.csv', index=False)
print("\n✅ Saved to z_predictions_simple.csv and publication figures")