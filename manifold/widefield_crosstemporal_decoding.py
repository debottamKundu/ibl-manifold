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

from manifold.widefield_decode import prepare_behavior

warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("crosstemporal.log")],
)
logger = logging.getLogger(__name__)

def cross_temporal_decode_cv(
    X_train_epoch, X_test_epoch, Y_train, Y_test, 
    apply_pca=False, n_components=None, 
    outer_cv_splits=5, inner_cv_splits=3, random_state=42
):
    """
    X_train_epoch: ntrials x features (Stimulus epoch)
    X_test_epoch: ntrials x features (Choice epoch)
    Y_train: ntrials (e.g., stim_side)
    Y_test: ntrials (e.g., choice)
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
               
        X_train, y_train_fold = X_train_epoch[train_idx], Y_train[train_idx]

        X_test, y_test_fold = X_test_epoch[test_idx], Y_test[test_idx]

        steps = [("scaler", RobustScaler())]
        
        if apply_pca:
            steps.append(("pca", PCA(n_components=n_components, random_state=random_state))) # type: ignore
            
        steps.append(("classifier", LogisticRegression(
            solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=random_state
        ))) # type: ignore

        pipeline = Pipeline(steps)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=1,
        )

        grid_search.fit(X_train, y_train_fold)
        best_model = grid_search.best_estimator_

        y_pred_probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

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

    # 3. LOAD NEURAL DATA
    data_stim, region_names = load_widefield_epoch(
        one, session_id, trials, config["hemisphere"], epoch="stim", response_time=True
    )
    data_choice, _ = load_widefield_epoch(
        one, session_id, trials, config["hemisphere"], epoch="choice", response_time=True
    )

    results = {}
    subepoch_train_idx = 0 
    subepoch_test_idx = 1  
    
    subepoch_key = f"train_stim{subepoch_train_idx}_test_choice{subepoch_test_idx}"
    results_subepoch = {}

    for region_data_stim, region_data_choice, region_name in tqdm(
        zip(data_stim, data_choice, region_names), desc="Regions", total=len(region_names)
    ):
        neural_train = region_data_stim[subepoch_train_idx, joint_mask, :]
        neural_test = region_data_choice[subepoch_test_idx, joint_mask, :]

        # Execute fitting with decoupled targets and PCA flag
        results_subepoch[region_name[0]] = fit_target_crosstemp(
            neural_train=neural_train, 
            neural_test=neural_test, 
            Y_train=target_stim,      # Train on Stim Side
            Y_test=target_choice,     # Test on Choice
            pseudo_train=pseudo_stim, 
            pseudo_test=pseudo_choice,
            apply_pca=apply_pca,
            n_components=n_components
        )

        region_mean = results_subepoch[region_name[0]]["mean_score"]
        region_null = np.nanmedian(results_subepoch[region_name[0]]["pseudosessions"])
        logger.info(
            f"Region: {region_name[0]} | Mean Score: {region_mean:.4f} | Median Null: {region_null:.4f}"
        )
        
    results[subepoch_key] = results_subepoch
    return results