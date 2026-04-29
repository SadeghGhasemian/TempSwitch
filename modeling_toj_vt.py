import warnings
warnings.filterwarnings("ignore")

import hddm
print(hddm.__version__)
import pandas as pd
import numpy as np
import os
import pickle
from joblib import Parallel, delayed

# ============================================
# Configuration
# ============================================
base_dir = r'E:\TemporalSwitch\HDDM_latest\power_analysis_toj_vt' # full model and different sample sizes
#base_dir = r'E:\TemporalSwitch\HDDM_latest\power_analysis_toj_vt\full_N35'
data_path = os.path.join(base_dir, 'tojDDM.csv')
models_base = os.path.join(base_dir, 'downsampling_models') # full model and different sample sizes
models_base = os.path.join(base_dir, 'models')
os.makedirs(models_base, exist_ok=True)

# ============================================
# Load data and preprocess (once)
# ============================================
print("Loading data...")
data = pd.read_csv(data_path)
all_participants = data['subj_idx'].unique()
print(f"Total participants: {len(all_participants)}")

# Preprocessing: multiply by 1000/28 and center
data["soa"] = data["soa"] * 100/28
mean_soa = data["soa"].mean()
data["soa"] = data["soa"] - mean_soa

# ============================================
# Regression model – now includes all parameters
# ============================================
reg_model = [
    "a ~ 1",                                            
    "v ~ soa + C(hist, Treatment('R')) + C(hist, Treatment('R')):soa", # full model and different sample sizes                                          
    "t ~ soa + C(hist, Treatment('R')) + C(hist, Treatment('R')):soa", # full model and different sample sizes                                         
    "z ~ 1"   
]

# ============================================
# Save function definition
# ============================================
def savePatch(self, fname):
    with open(fname, 'wb') as f:
        pickle.dump(self, f)

# ============================================
# Function to run one chain for a given (n_subj, chain_id, participant list)
# ============================================
def run_chain(n_subj, chain_id, participant_list):
    """Fit one chain on the fixed participant subset."""
    hddm.HDDM.savePatch = savePatch

    # Seed for MCMC (different per chain)
    np.random.seed(chain_id * 1000 + n_subj)

    subset = data[data['subj_idx'].isin(participant_list)].copy()

    m = hddm.HDDMRegressor(
        subset, reg_model,
        group_only_regressors=True,
        include={'z'}          # still correct – only z has predictors
    )

    m.find_starting_values()

    # Database and model file paths
    chain_dir = os.path.join(models_base, f'N{n_subj}')
    os.makedirs(chain_dir, exist_ok=True)
    dbname = os.path.join(chain_dir, f'chain{chain_id}.db')
    fname = os.path.join(chain_dir, f'chain{chain_id}')

    m.sample(4000, burn=2000, thin=5, dbname=dbname, db='pickle')
    m.savePatch(fname)
    print(f"  Saved: N={n_subj}, chain={chain_id}")
    return True

# ============================================
# Main loop over sample sizes
# ============================================
sample_sizes = [10, 15, 20, 25, 30, 35] # full model and different sample sizes                                          
n_chains = 4

print("Starting downsampling generation (no repetitions)...")
for n in sample_sizes:
    print(f"\nProcessing N = {n}")
    # Generate one random subset of participants (fixed seed for all chains)
    np.random.seed(n * 1000)
    chosen = np.random.choice(all_participants, size=n, replace=False)

    # Run 4 chains in parallel on the same participant subset
    Parallel(n_jobs=n_chains)(
        delayed(run_chain)(n, ch, chosen) for ch in range(n_chains)
    )

print("\nAll models saved.")