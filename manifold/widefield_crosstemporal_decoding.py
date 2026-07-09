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
from sklearn.decomposition import PCA
import warnings

from manifold.widefield_decode import prepare_behavior
from manifold.widefield_ppi import aggregate_by_parent, beryl_mapping

warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("crosstemporal.log")],
)
logger = logging.getLogger(__name__)
config = check_config_decoding()

def cross_temporal_decode_cv(
    X_train_epoch, X_test_epoch, Y_train, Y_test, 
    n_components=5, outer_cv_splits=5, inner_cv_splits=3, random_state=42
):
    """
    X_train_epoch: ntrials x N_features (Stimulus Epoch)
    X_test_epoch: ntrials x M_features (Choice Epoch)
    """
    outer_cv = StratifiedKFold(n_splits=outer_cv_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(len(Y_test), dtype=float)
    
    param_grid = {
        "classifier__C": np.logspace(-2, 2, 5),
        "classifier__penalty": ["l2"],
    }
    outer_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_train_epoch, Y_train)):
        

        X_train_fold_A = X_train_epoch[train_idx]
        y_train_fold = Y_train[train_idx]

        X_train_fold_B = X_test_epoch[train_idx] 
        X_test_fold_B = X_test_epoch[test_idx]
        y_test_fold = Y_test[test_idx]

        scaler_A = RobustScaler()
        pca_A = PCA(n_components=n_components, random_state=random_state)
        
        X_train_pca = pca_A.fit_transform(scaler_A.fit_transform(X_train_fold_A))

        scaler_B = RobustScaler()
        pca_B = PCA(n_components=n_components, random_state=random_state)
        
        pca_B.fit(scaler_B.fit_transform(X_train_fold_B))
        X_test_pca = pca_B.transform(scaler_B.transform(X_test_fold_B))

        pipeline = Pipeline([
            ("classifier", LogisticRegression(
                solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=random_state
            ))
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=1,
        )

        grid_search.fit(X_train_pca, y_train_fold)
        best_model = grid_search.best_estimator_


        y_pred_probs = best_model.predict_proba(X_test_pca)[:, 1]
        y_pred = best_model.predict(X_test_pca)

        oof_predictions[test_idx] = y_pred_probs
        fold_score = balanced_accuracy_score(y_test_fold, y_pred)

        outer_scores.append(fold_score)

    return {
        "oof_predictions": oof_predictions,
        "outer_scores": outer_scores,
        "mean_score": np.mean(outer_scores),
        "std_score": np.std(outer_scores),
    }

def run_single_animal_crosstemp(session_id, apply_pca=False, n_components=None):
    logger.info(f"--- Starting Cross-Temporal session: {session_id} ---")

    one = ONE(mode="local")
    ssl = SessionLoader(one, session_id)
    ssl.load_trials(collection="alf")
    trials = ssl.trials.copy()

    out = np.nan_to_num(trials.contrastLeft) - np.nan_to_num(trials.contrastRight)
    trials["signcont"] = out
    trials["stim_side"] = np.sign(trials["signcont"])

    _, stim_mask = load_trials_and_mask(one, session_id, exclude_nochoice=False, exclude_unbiased=True)
    stimulus_non_zero = trials["signcont"] != 0
    stim_mask = stim_mask & stimulus_non_zero
    
    _, choice_mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=True)
    
    joint_mask = stim_mask & choice_mask
    logger.info(f"Joint mask applied. {joint_mask.sum()} valid trials remaining.")


    target_stim, _ = prepare_behavior(
        trials, session_id, joint_mask, epoch="stim",pseudosessions=1
    )
    target_choice, _ = prepare_behavior(
        trials, session_id, joint_mask, epoch="choice",pseudosessions=1
    )

    stim_regions = ["MOB","MOs","MOp"]
    choice_regions = ["PL"]

    stim_regions =[ [x] for x in stim_regions]
    choice_regions = [[x] for x in choice_regions]

    stim_data, stim_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="stim", regions=stim_regions)
    choice_data, choice_region_names = load_widefield_epoch(one, session_id, trials, config["hemisphere"], epoch="choice", response_time=True, regions=choice_regions)

    parent_mapping = beryl_mapping()
    stim_data, stim_region_names = aggregate_by_parent(stim_data, stim_region_names, parent_mapping)
    choice_data, choice_region_names = aggregate_by_parent(choice_data, choice_region_names, parent_mapping)

    results = {}

    for region_stim_idx in range(len(stim_data)):
        stim_region = stim_data[region_stim_idx]
        stim_region = stim_region[1,:]-stim_region[0,:]

        for region_choice_idx in range(len(choice_data)): # should be 1
            choice_region = choice_data[region_choice_idx]

            resultx = cross_temporal_decode_cv(stim_region, choice_region, target_stim, target_choice, apply_pca=False)
            key = f'{stim_region_names[region_stim_idx]}, {choice_region_names[region_choice_idx]}'
            results[key] = resultx
    return results


# now we can run this for an animal, and the regions we want, given the decoders are significant in a particular eid
