import numpy as np
import pandas as pd
import pickle as pkl
import logging
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import os
import pathlib

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import load_trials_and_mask
from communication_subspace.ibl_communication.utils import load_widefield_epoch
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

from manifold.decoding.functions.utils import check_config_decoding
from manifold.widefield_ppi import aggregate_by_parent, beryl_mapping

config = check_config_decoding()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("spatial_alignment_debug.log")],
)
logger = logging.getLogger(__name__)


def process_single_animal(session_id, significant_pickles, n_splits=5):
    one = ONE(mode="local")
    ssl = SessionLoader(one, session_id)
    ssl.load_trials(collection="alf")
    trials = ssl.trials.copy()

    # Determine stimulus side
    out = np.nan_to_num(trials.contrastLeft) - np.nan_to_num(trials.contrastRight)
    trials["signcont"] = out
    trials["stim_side"] = np.sign(trials["signcont"])

    # Load masks
    _, stimulus_behavioral_mask = load_trials_and_mask(
            one, session_id, exclude_nochoice=False, exclude_unbiased=False)
    stimulus_non_zero_mask = trials["signcont"] != 0
    stim_mask = stimulus_behavioral_mask & stimulus_non_zero_mask
    
    _, choice_behavioral_mask = load_trials_and_mask(
            one, session_id, exclude_nochoice=True, exclude_unbiased=False)
    choice_mask = choice_behavioral_mask
    combined_mask = choice_mask & stim_mask

    correct_mask = trials["feedbackType"] == 1
    incorrect_mask = trials["feedbackType"] == -1

    final_correct_mask = combined_mask & correct_mask
    final_incorrect_mask = combined_mask & incorrect_mask

    correct_indices = np.where(final_correct_mask)[0]
    incorrect_indices = np.where(final_incorrect_mask)[0]
    
    if len(correct_indices) < n_splits:
        logger.warning(f"Skipping {session_id} - not enough correct trials for {n_splits}-fold CV.")
        return None

    if session_id not in significant_pickles:
        logger.warning(f"Session {session_id} not in significant_pickles.")
        return None
        
    stim_regions = [[x] for x in significant_pickles[session_id].get('stim', [])]
    choice_regions = [[x] for x in significant_pickles[session_id].get('choice', [])]

    if len(stim_regions) == 0 and len(choice_regions) == 0:
        logger.warning(f"Skipping {session_id} - no significant regions.")
        return None

    logger.info("Loading widefield epoch data...")
    stim_data, stim_region_names = [], []
    choice_data, choice_region_names = [], []
    
    if len(stim_regions) > 0:
        stim_data, stim_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="stim", regions=stim_regions)
    if len(choice_regions) > 0:
        choice_data, choice_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="choice", response_time=True, regions=choice_regions)

    parent_mapping = beryl_mapping()
    if len(stim_data) > 0:
        stim_data, stim_region_names = aggregate_by_parent(stim_data, stim_region_names, parent_mapping)
    if len(choice_data) > 0:
        choice_data, choice_region_names = aggregate_by_parent(choice_data, choice_region_names, parent_mapping)

    labels = trials["stim_side"].values

    results = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    correct_labels = labels[correct_indices]

    def process_epoch_regions(data, region_names, epoch_name):
        for idx in range(len(data)):
            region_name = region_names[idx]
            if epoch_name == 'stim':
                frame = data[idx][1, :] - data[idx][0, :]
            else:
                frame = data[idx][1, :]
            
            X_correct = frame[correct_indices, :]
            X_incorrect = frame[incorrect_indices, :] if len(incorrect_indices) > 0 else None
            y_correct = correct_labels
            
            oof_logits = np.zeros(len(correct_indices))
            
            models = []
            
            for train_idx, test_idx in skf.split(X_correct, y_correct):
                X_train, y_train = X_correct[train_idx], y_correct[train_idx]
                X_test, y_test = X_correct[test_idx], y_correct[test_idx]
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegressionCV(cv=3, random_state=42, max_iter=1000, n_jobs=-1))
                ])
                pipeline.fit(X_train, y_train)
                
                oof_logits[test_idx] = pipeline.decision_function(X_test)
                models.append(pipeline)
                
            incorrect_logits = None
            if X_incorrect is not None:
                incorrect_logits_all_folds = [m.decision_function(X_incorrect) for m in models]
                incorrect_logits = np.mean(incorrect_logits_all_folds, axis=0)
                
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            oof_preds = np.where(oof_logits > 0, 1, -1)
            cv_acc = accuracy_score(y_correct, oof_preds)
            cv_b_acc = balanced_accuracy_score(y_correct, oof_preds)
                
            results.append({
                "region": region_name,
                "epoch": epoch_name,
                "cv_accuracy": cv_acc,
                "cv_balanced_accuracy": cv_b_acc,
                "correct_trial_indices": correct_indices,
                "correct_logits": oof_logits,
                "correct_true_labels": y_correct,
                "incorrect_trial_indices": incorrect_indices,
                "incorrect_logits": incorrect_logits,
                "incorrect_true_labels": labels[incorrect_indices] if len(incorrect_indices) > 0 else None
            })

    if len(stim_data) > 0:
        logger.info("Processing stim regions...")
        process_epoch_regions(stim_data, stim_region_names, 'stim')
        
    if len(choice_data) > 0:
        logger.info("Processing choice regions...")
        process_epoch_regions(choice_data, choice_region_names, 'choice')
        
    return results

def process_session_epoch(session, significantregions, output_dir):
    try:
        results = process_single_animal(session, significantregions)
        if results is None:
            return False
            
        output_path = os.path.join(output_dir, f"{session}_spatial_alignment_logits.pkl")
        
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
    import sys
    logger.info("Initializing ONE and searching for datasets...")
    one = ONE(mode="local")
    sessions_all = one.search(datasets="widefieldU.images.npy")
    sessions_all = np.asarray([str(s) for s in sessions_all])
    logger.info(f"Found {len(sessions_all)} sessions with widefield data.")

    output_dir = "./data/generated/wifi/spatial_alignment"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open("./data/processed/significant_stims_choice.pkl",'rb') as f:
        significant_preloads = pkl.load(f)

    # For testing, if an argument is passed, test just that session
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_session = sessions_all[0]
        logger.info(f"Running test on single session: {test_session}")
        process_session_epoch(test_session, significant_preloads, output_dir)
        sys.exit(0)
    
    total_tasks = len(sessions_all)
    successful_tasks = 0

    logger.info(f"Starting serial processing for {total_tasks} tasks...")

    for session in sessions_all:
        try:
            success = process_session_epoch(session, significant_preloads, output_dir)
            if success:
                successful_tasks += 1
        except Exception as e:
            logger.error(f"Task {session} generated an unexpected exception: {e}")

    logger.info(
        f"Processing complete! Successfully processed {successful_tasks}/{total_tasks} tasks."
    )
