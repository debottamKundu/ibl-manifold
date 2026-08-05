from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import logging
from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from collections import defaultdict
import pandas as pd
from manifold.decoding.functions.utils import check_config_decoding
import numpy as np
import pickle as pkl
from manifold.decoding.functions import nulldistributions
from communication_subspace.ibl_communication.utils import load_widefield_epoch
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
from manifold.utils import get_trial_masks
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import statsmodels.stats.multitest as smm
import seaborn as sns


def compute_group_level_ppi(combined_df, min_animals=3):
    subject_means = combined_df.groupby(
        ['animal-id', 'condition', 'seed', 'target']
    )['interaction_beta'].mean().reset_index()
    results = []
    
    
    grouped = subject_means.groupby(['condition', 'seed', 'target'])
    
    for (condition, seed, target), group_data in grouped:
        
    
        betas = group_data['interaction_beta'].dropna().values
        
        n_animals = len(betas)
        
    
        if n_animals < min_animals:
            continue
            
    
        t_stat, p_val = ttest_1samp(betas, popmean=0)
        mean_beta = np.mean(betas)
        sem_beta = np.std(betas, ddof=1) / np.sqrt(n_animals)
        
        results.append({
            'condition': condition,
            'seed': seed,
            'target': target,
            'mean_beta': mean_beta,
            'sem_beta': sem_beta,
            't_stat': t_stat,
            'p_val_raw': p_val,
            'n_animals': n_animals
        })
        
    results_df = pd.DataFrame(results)
    final_dfs = []
    
    for condition in results_df['condition'].unique():
        cond_df = results_df[results_df['condition'] == condition].copy()
        
        reject, pvals_corrected, _, _ = smm.multipletests(
            cond_df['p_val_raw'], alpha=0.05, method='fdr_bh'
        )
        
        cond_df['p_val_fdr'] = pvals_corrected
        cond_df['is_significant'] = reject
        final_dfs.append(cond_df)
        
    final_results = pd.concat(final_dfs, ignore_index=True)
    final_results = final_results.sort_values(by=['condition', 'p_val_fdr'])
    
    return final_results


def compute_complete_df(files):
    
    df = []
    for idx,f in enumerate(files):
        data_session = pkl.load(open(f,'rb'))
        data_session['animal-id'] = idx
        df.append(data_session)
    combined_df = pd.concat(df)
    # combined_df['seed'] = combined_df['seed']
    # combined_df['target'] = combined_df['target']

    return combined_df
