import logging
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import os

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import load_trials_and_mask
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas
from communication_subspace.ibl_communication.utils import load_widefield_epoch
from manifold.utils import get_trial_masks
from manifold.widefield_ppi import beryl_mapping, aggregate_by_parent, return_labels
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = {
    "hemisphere": ("left", "right")
}

def compute_cca_score(Y, X, pca_n=10):
    """
    Computes CCA between X and Y after reducing them to pca_n components.
    X, Y: shape (n_trials, n_voxels)
    Returns: Canonical correlation coefficient (Pearson r between 1st canonical variates)
    """
    if X.shape[0] < pca_n:
        pca_n = X.shape[0] - 1
        
    if pca_n <= 0:
         return np.nan

    pca_x = PCA(n_components=pca_n)
    pca_y = PCA(n_components=pca_n)
    
    try:
        X_reduced = pca_x.fit_transform(X)
        Y_reduced = pca_y.fit_transform(Y)
        
        cca = CCA(n_components=1)
        X_c, Y_c = cca.fit_transform(X_reduced, Y_reduced)
        
        # Pearson correlation between the first canonical variates
        r, _ = pearsonr(X_c[:, 0], Y_c[:, 0])
        return r
    except Exception as e:
        logger.warning(f"CCA failed: {e}")
        return np.nan


def process_single_animal(session_id, significant_pickles, n_iterations=10):
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

    if session_id not in significant_pickles:
        logger.warning(f"Session {session_id} not in significant_pickles.")
        return pd.DataFrame([])

    stim_regions = significant_pickles[session_id]['stim']
    choice_regions = significant_pickles[session_id]['choice']

    stim_regions =[ [x] for x in stim_regions]
    choice_regions = [[x] for x in choice_regions]
    logger.info("Loading widefield epoch data...")
    
    try:
        stim_data, stim_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="stim", regions=stim_regions)
        choice_data, choice_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="choice", response_time=True, regions=choice_regions)
    except Exception as e:
        logger.error(f"Failed loading widefield epoch for {session_id}: {e}")
        return pd.DataFrame([])

    parent_mapping = beryl_mapping()
    stim_data, stim_region_names = aggregate_by_parent(stim_data, stim_region_names, parent_mapping)
    choice_data, choice_region_names = aggregate_by_parent(choice_data, choice_region_names, parent_mapping)

    session_results = []
    labels, conditions = return_labels(trials, "congruence")
    final_mask = combined_mask & conditions

    valid_indices = np.where(final_mask)[0]
    idx_cond_a = [i for i in valid_indices if labels[i] == 1]
    idx_cond_b = [i for i in valid_indices if labels[i] == -1]
    min_trials = min(len(idx_cond_a), len(idx_cond_b))

    if min_trials < 10:
        logger.warning(f"Skipping for {session_id} - {min_trials} min trials is too low for CCA.")
        return pd.DataFrame([])
            
    for iter_idx in range(n_iterations):
        sub_a = np.random.choice(idx_cond_a, min_trials, replace=False)
        sub_b = np.random.choice(idx_cond_b, min_trials, replace=False)
        
        for idx in tqdm(range(len(stim_data)), desc=f'stim frames - {"congruence"} - iter {iter_idx}', leave=False):
            # stim is frame 1 - frame 0
            stim_frame_a = stim_data[idx][:, sub_a, :]
            stim_frame_a = stim_frame_a[1, :] - stim_frame_a[0, :]
            
            stim_frame_b = stim_data[idx][:, sub_b, :]
            stim_frame_b = stim_frame_b[1, :] - stim_frame_b[0, :]
            
            stim_region_name = stim_region_names[idx]
            
            for idy in range(len(choice_data)):
                # choice is frame 1
                choice_frame_a = choice_data[idy][:, sub_a, :]
                choice_frame_a = choice_frame_a[1, :]
                
                choice_frame_b = choice_data[idy][:, sub_b, :]
                choice_frame_b = choice_frame_b[1, :]
                
                choice_region_name = choice_region_names[idy]
                
                cca_congruent = compute_cca_score(choice_frame_a, stim_frame_a, pca_n=10)
                cca_incongruent = compute_cca_score(choice_frame_b, stim_frame_b, pca_n=10)
                
                session_results.append({
                    "condition": "congruence",
                    "seed": stim_region_name,
                    "target": choice_region_name,
                    "cca_congruent": cca_congruent,
                    "cca_incongruent": cca_incongruent,
                    "n_trials": min_trials,
                    "iteration": iter_idx
                })
                
    return pd.DataFrame(session_results)    
    

def process_session_epoch(session, significantregions):
    try:
        results = process_single_animal(session, significantregions)
        if results.empty:
            return False
            
        output_dir = "./data/generated/wifi/ccas/deltaframe"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{session}_cca_significant_regions_pca_aggregated.pkl"

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

    total_tasks = len(sessions_all)
    successful_tasks = 0

    logger.info(f"Starting sequential processing for {total_tasks} tasks...")

    # Load significant regions dictionary mapping session_id -> {'stim': [...], 'choice': [...]}
    significant_preloads_path = "./data/processed/significant_stims_choice.pkl"
    if os.path.exists(significant_preloads_path):
        with open(significant_preloads_path, 'rb') as f:
            significant_preloads = pkl.load(f)
    else:
        logger.error(f"Could not find {significant_preloads_path}")
        exit(1)

    success = process_session_epoch(sessions_all[0], significant_preloads)
    # for session in sessions_all:
    #     try:
    #         success = process_session_epoch(session, significant_preloads)
    #         if success:
    #             successful_tasks += 1
    #     except Exception as e:
    #         logger.error(f"Task {(session)} generated an unexpected exception: {e}")

    # logger.info(
    #     f"Processing complete! Successfully processed {successful_tasks}/{total_tasks} tasks."
    # )
