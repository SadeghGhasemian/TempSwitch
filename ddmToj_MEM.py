
############################################# model ###########################################
# # run_model.py
import warnings
warnings.filterwarnings("ignore")

import hddm
import pickle
from kabuki.analyze import gelman_rubin
from kabuki.utils import concat_models
import os
import pandas as pd
from joblib import Parallel, delayed

# --- Config ---
data_path = r'E:\TemporalSwitch\HDDM _new\Drift_TOJ\tojDDM.csv'
modelName = 'MEM_toj'
data = hddm.load_csv(data_path)
data_folder = os.path.dirname(data_path)
models_dir = os.path.join(data_folder, 'MEM_Models')
results_dir = os.path.join(data_folder, 'MEM_Results')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

reg_model = [
    "v ~ soa + C(hist, Treatment('R'))",
    "t ~ soa + C(hist, Treatment('R'))"
]

def savePatch(self, fname):
    with open(fname, 'wb') as f:
        pickle.dump(self, f)


def run_model(i):
    hddm.HDDM.savePatch = savePatch
    m = hddm.HDDMRegressor(
        data, reg_model,
        group_only_regressors=True,
        include={"st"}
    )

    dbname = os.path.join(models_dir, f'{modelName}_{i}.db')
    fname = os.path.join(models_dir, f'{modelName}_{i}')
    #m.find_starting_values()
    m.sample(3000, burn=1500, thin=5, dbname=dbname, db='pickle')
    m.savePatch(fname)
    return m

if __name__ == "__main__":
    num_jobs = 4
    models = Parallel(n_jobs=num_jobs)(delayed(run_model)(i) for i in range(num_jobs))

    combined_model = concat_models(models)
    stats = gelman_rubin(models)
    pd.DataFrame.from_dict(stats, orient='index').to_csv(os.path.join(results_dir, f'{modelName}_RHat.csv'))
    combined_model.gen_stats().to_csv(os.path.join(results_dir, f'stats_{modelName}.csv'))
    combined_model.get_traces().to_csv(os.path.join(results_dir, f'{modelName}_traces.csv'))
    pd.DataFrame({'DIC': [combined_model.dic]}).to_csv(os.path.join(results_dir, f'{modelName}_DIC.csv'), index=False)

########################################## plot #############################
# ##############################################################################    

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import hddm
import matplotlib
matplotlib.use('Agg')
from kabuki.utils import concat_models

# --- Config ---
modelName = 'MEM_toj'
data_path = r'E:\TemporalSwitch\HDDM _new\Drift_TOJ\tojDDM.csv'
data_folder = os.path.dirname(data_path)
models_dir = os.path.join(data_folder, 'MEM_Models')
plots_dir = os.path.join(data_folder, 'MEM_Plots', modelName)
os.makedirs(plots_dir, exist_ok=True)

# --- Reload Model ---
num_jobs = 4
models = []
for i in range(num_jobs):
    with open(os.path.join(models_dir, f'{modelName}_{i}'), 'rb') as f:
        models.append(pickle.load(f))

m_comb = concat_models(models)

# --- Temporarily change directory to save plots there ---
original_dir = os.getcwd()
os.chdir(plots_dir)

# Save all posterior plots as PNGs in the plots directory
m_comb.plot_posteriors(save=True)

# Change back to original directory
os.chdir(original_dir)


##################################################### PPC #######################################
#################################################################################################
# ppc_check.py
import warnings
warnings.filterwarnings("ignore")

import pickle
import hddm
import pandas as pd
import os
import numpy as np
import pymc as pm
import pymc.progressbar as pbar
from kabuki.utils import concat_models

# --- Config ---
modelName = 'MEM_toj'
data_path = r'E:\TemporalSwitch\HDDM _new\Drift_TOJ\tojDDM.csv'
data = hddm.load_csv(data_path)
data_folder = os.path.dirname(data_path)
models_dir = os.path.join(data_folder, 'MEM_Models')
results_dir = os.path.join(data_folder, 'MEM_Results')
ppc_dir = os.path.join(data_folder, 'MEM_PPC')
os.makedirs(ppc_dir, exist_ok=True)

# --- Reload Model ---
num_jobs = 4
reloaded_models = []
for i in range(num_jobs):
    with open(os.path.join(models_dir, f'{modelName}_{i}'), 'rb') as f:
        reloaded_models.append(pickle.load(f))

m_comb = concat_models(reloaded_models)

# --- PPC Functions ---
def _parents_to_random_posterior_sample(bottom_node, pos=None):
    for parent in bottom_node.extended_parents:
        if isinstance(parent, pm.Node):
            if pos is None:
                pos = np.random.randint(0, len(parent.trace()))
            parent.value = parent.trace()[pos]

def _post_pred_generate(bottom_node, samples=100, data=None, append_data=True):
    datasets = []
    for _ in range(samples):
        _parents_to_random_posterior_sample(bottom_node)
        sampled = bottom_node.random()
        sampled.reset_index(inplace=True)
        if append_data and data is not None:
            sampled = sampled.join(data.reset_index(), lsuffix='_sampled')
        datasets.append(sampled)
    return datasets

def post_pred_gen(model, groupby=None, samples=100, append_data=False, progress_bar=True):
    results = {}
    if progress_bar:
        bar = pbar.progress_bar(len(model.get_observeds()))
    if groupby is None:
        iter_data = ((name, model.data.loc[obs['node'].value.index]) for name, obs in model.iter_observeds())
    else:
        iter_data = model.data.groupby(groupby)

    i = 0
    for name, data in iter_data:
        node = model.get_data_nodes(data.index)
        if node is None or not hasattr(node, 'random'):
            continue
        datasets = _post_pred_generate(node, samples, data, append_data)
        results[name] = pd.concat(datasets, names=['sample'], keys=list(range(len(datasets))))
        if progress_bar:
            bar.update(i + 1)
        i += 1
    return pd.concat(results, names=['node'])

# --- Generate and Save PPC Data ---
ppc_data = post_pred_gen(m_comb, samples=100, append_data=True)
ppc_data.to_csv(os.path.join(ppc_dir, f'{modelName}_simData.csv'))

# --- Save PPC Comparison ---
ppc_compare = hddm.utils.post_pred_stats(data, ppc_data)
ppc_compare.to_csv(os.path.join(results_dir, 'PPC_compare.csv'))
print(ppc_compare)
