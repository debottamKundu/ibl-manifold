# for regions with good enough decoding
# we run ppi analysis


import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
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

warnings.filterwarnings("ignore")
from manifold.utils import get_trial_masks
config = check_config_decoding()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("decoding_pipeline_debug.log")],
)
logger = logging.getLogger(__name__)

def reduce_dimensions(data, method=None):
    # data is in trials x voxels
    if method=='PCA':
        pca = PCA(n_components=1) # akin to SPM
        return pca.fit_transform(data)
    elif method=='mean':
        return np.mean(data, axis=1, keepdims=True)
    else:
        return data


def compute_ppi_interaction(Y, X, labels, reduction='mean'):

    X = reduce_dimensions(X, method=reduction)
    Y = reduce_dimensions(Y, method=reduction)

    x1 = X.squeeze()
    # mean center?
    x1 = x1-np.mean(x1)
    labels_centered = labels - np.mean(labels)
    interaction = labels_centered*x1

    X_mat = np.column_stack((np.ones(len(labels)), x1, labels_centered, interaction))

    if Y.shape[1] == 1:
        model = sm.OLS(Y.squeeze(), X_mat)
        results = model.fit()
        return results
    else:
        betas, residuals, rank, s = np.linalg.lstsq(X_mat, Y, rcond=None)
        return betas

def return_labels(trials, condition):
    masks, _ = get_trial_masks(trials)
    n_trials = len(masks['Congruent_correct'])
    labels = np.zeros(n_trials)

    if condition == 'congruence':
        conda = masks['Congruent_correct'] | masks['Congruent_incorrect']
        condb = masks['Incongruent_correct'] | masks['Incongruent_incorrect']       

    elif condition == 'correctness':
        conda = masks['Incongruent_correct']
        condb = masks['Incongruent_incorrect']

    labels[conda] = 1
    labels[condb] = -1 

    valid_mask = conda|condb # type: ignore
    return labels, valid_mask

def process_single_animal(session_id, significant_pickles):

    one = ONE(mode="local")
    ssl = SessionLoader(one, session_id)
    ssl.load_trials(collection="alf")
    trials = ssl.trials.copy()

    out = np.nan_to_num(trials.contrastLeft) - np.nan_to_num(trials.contrastRight)
    trials["signcont"] = out
    trials["stim_side"] = np.sign(trials["signcont"])

    _, stimulus_behavioral_mask = load_trials_and_mask(
            one,
            session_id,
            exclude_nochoice=False,
            exclude_unbiased=False,
        )
    stimulus_non_zero_mask = trials["signcont"] != 0
    stim_mask = stimulus_behavioral_mask & stimulus_non_zero_mask
    _, choice_behavioral_mask = load_trials_and_mask(
            one,
            session_id,
            exclude_nochoice=True,
            exclude_unbiased=False,
        )
    choice_mask = choice_behavioral_mask
    combined_mask = choice_mask & stim_mask

    stim_regions = significant_pickles[session_id]['stim']
    choice_regions = significant_pickles[session_id]['choice']

    stim_regions =[ [x] for x in stim_regions]
    choice_regions = [[x] for x in choice_regions]
    logger.info("Loading widefield epoch data...")
    
    stim_data, stim_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="stim", regions=stim_regions)
    choice_data, choice_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="choice", response_time=True, regions=choice_regions)

    # stim is frame2 - frame1
    # choice is frame2
    session_results = []
    for terms in ["correctness","congruence"]:  
        labels, conditions = return_labels(trials, terms)
        final_mask = combined_mask & conditions
        labels_masked = labels[final_mask]

        # compute difference between stim frames
        # get frame 1 for choice
        # run ppi
        for idx in tqdm(range(len(stim_data)),desc='stim frames'):
            stim_frame = stim_data[idx][:, final_mask, :]
            stim_frame = stim_frame[1,:] - stim_frame[0,:] # do the delta 
            stim_region_name = stim_region_names[idx]
            for idy in range(len(choice_data)):
                choice_frame = choice_data[idy][:, final_mask, :]
                choice_frame = choice_frame[1,:]
                choice_region_name = choice_region_names[idy]
                results = compute_ppi_interaction(Y=choice_frame, X=stim_frame, labels=labels_masked, reduction="PCA")
                session_results.append({
                    'interaction_beta':results.params[3],# type: ignore
                    "condition":terms,
                    "seed": stim_region_name,
                    "target":choice_region_name,
                    "n_trials":len(labels_masked)
                    }) 
    # now what
    # 
    return pd.DataFrame(session_results)    
    


def process_session_epoch(session, significantregions):
    try:
        #   print(session, subepoch)
        # fast execution, we treat change as null
        results = process_single_animal(session, significantregions)
        output_path = f"./data/generated/wifi/ppis/{session}_ppi_significant_regions_pca.pkl"

        with open(output_path, "wb") as f:
            pkl.dump(results, f)
            logger.info(f"Successfully saved results to {output_path}")
        return True

    except Exception as e:
        logger.error(
            f"Failed processing session {session}. Error: {e}",
            exc_info=True,
        )
        return False


if __name__ == "__main__":
    logger.info("Initializing ONE and searching for datasets...")
    one = ONE(mode="local")
    sessions_all = one.search(datasets="widefieldU.images.npy")
    sessions_all = np.asarray([str(s) for s in sessions_all])  # type: ignore
    logger.info(f"Found {len(sessions_all)} sessions with widefield data.")

 

    # subepochs = ["stim-both"]
    
    total_tasks = len(sessions_all)
    successful_tasks = 0

    logger.info(f"Starting parallel processing for {total_tasks} tasks...")


    with open("./data/processed/significant_stims_choice.pkl",'rb') as f:
        significant_preloads = pkl.load(f)

    # results = process_session_epoch(sessions_all[0], significant_preloads)

    for session in sessions_all:
        try:
            success = process_session_epoch(session, significant_preloads)
            if success:
                successful_tasks += 1
        except Exception as e:
            logger.error(f"Task {(session)} generated an unexpected exception: {e}")

    logger.info(
        f"Processing complete! Successfully processed {successful_tasks}/{total_tasks} tasks."
    )