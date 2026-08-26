import os
import logging
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

from one.api import ONE
from brainbox.io.one import SessionLoader
from tqdm import tqdm
from brainwidemap import load_trials_and_mask
from communication_subspace.ibl_communication.utils import prepare_widefield
from manifold.widefield_ppi import aggregate_by_parent, beryl_mapping
from manifold.decoding.functions.utils import check_config_decoding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config = check_config_decoding()

def compute_congruency(trials):
    cont_left = np.nan_to_num(trials['contrastLeft'].values)
    cont_right = np.nan_to_num(trials['contrastRight'].values)
    signcont = cont_left - cont_right
    stim_side = np.sign(signcont)
    
    prob_left = trials['probabilityLeft'].values
    expected_side = np.sign(prob_left - 0.5)
    
    is_congruent = (stim_side == expected_side) & (expected_side != 0)
    
    trials['stim_side_computed'] = stim_side
    trials['signed_contrast'] = signcont
    trials['is_congruent'] = is_congruent
    
    return trials

def load_temporal_widefield_epoch(one, session_id, trials, hemisphere, epoch, regions, aggregate_parent=False):
    if epoch == "stim":
        align_times = trials.stimOn_times
        frame_window = [0, 4] 
    elif epoch == "choice":
        align_times = trials.response_times
        frame_window = [-4, 0]
    else:
        raise ValueError("epoch must be stim or choice")
        
    data_epoch, actual_regions = prepare_widefield(
        one,
        session_id,
        hemisphere,
        regions=[[r] for r in regions],
        align_times=align_times,
        frame_window=frame_window,
        functional_channel=470,
        stage_only=False,
    )
    
    data_epoch_reduced = []
    region_names = []

    for idx in range(len(data_epoch)):
        n_voxels = data_epoch[idx].shape[-1]
        if n_voxels < 5:
            continue
        data_epoch_reduced.append(data_epoch[idx].transpose(1, 0, 2))
        region_names.append(actual_regions[idx])
        
    if aggregate_parent:
        parent_mapping = beryl_mapping()
        data_epoch_reduced, region_names = aggregate_by_parent(data_epoch_reduced, region_names, parent_mapping)
        
    return data_epoch_reduced, region_names

def train_and_project_temporal_nested(X_correct_all_frames, y_correct, X_incorrect_all_frames, train_frame_idx, classifier_name='svc'):
    n_frames = X_correct_all_frames.shape[0]
    n_correct = y_correct.shape[0]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if classifier_name == 'svc':
        base_model = LinearSVC(penalty='l2', dual=False, max_iter=10000, random_state=42)
        param_grid = {'clf__C': np.logspace(-3, 2, 6)}
    elif classifier_name == 'ridge':
        base_model = RidgeClassifier(random_state=42)
        param_grid = {'clf__alpha': np.logspace(-3, 2, 6)}
        
    X_train_target = X_correct_all_frames[train_frame_idx]
    
    corr_projections_by_frame = [np.zeros(n_correct) for _ in range(n_frames)]
    inc_logits_folds = [[] for _ in range(n_frames)]
    models = []
    
    oof_target_preds = np.zeros(n_correct)
    
    for train_idx, test_idx in skf.split(X_train_target, y_correct):
        X_train, y_train = X_train_target[train_idx], y_correct[train_idx]
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', base_model)
        ])
        
        grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='balanced_accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        oof_target_preds[test_idx] = best_model.predict(X_train_target[test_idx])
        
        for f in range(n_frames):
            corr_projections_by_frame[f][test_idx] = best_model.decision_function(X_correct_all_frames[f][test_idx])
            if X_incorrect_all_frames is not None and len(X_incorrect_all_frames[0]) > 0:
                inc_logits_folds[f].append(best_model.decision_function(X_incorrect_all_frames[f]))
                
        models.append({'weights': best_model.named_steps['clf'].coef_, 'best_params': grid.best_params_})
        
    overall_bacc = balanced_accuracy_score(y_correct, oof_target_preds)
    
    inc_projections_by_frame = []
    if X_incorrect_all_frames is not None and len(X_incorrect_all_frames[0]) > 0:
        for f in range(n_frames):
            inc_projections_by_frame.append(np.mean(inc_logits_folds[f], axis=0))
            
    model_info = {
        'models': models,
        'overall_bacc': overall_bacc,
        'trained_frame_idx': train_frame_idx
    }
    
    return corr_projections_by_frame, inc_projections_by_frame, model_info

def train_and_project_temporal_single(X_correct_all_frames, y_correct, X_incorrect_all_frames, train_frame_idx, classifier_name='svc'):
    n_frames = X_correct_all_frames.shape[0]
    
    if classifier_name == 'svc':
        base_model = LinearSVC(penalty='l2', dual=False, max_iter=10000, random_state=42)
        param_grid = {'clf__C': np.logspace(-3, 2, 6)}
    elif classifier_name == 'ridge':
        base_model = RidgeClassifier(random_state=42)
        param_grid = {'clf__alpha': np.logspace(-3, 2, 6)}
        
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', base_model)
    ])
    
    X_train_target = X_correct_all_frames[train_frame_idx]
    
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='balanced_accuracy')
    grid.fit(X_train_target, y_correct)
    
    best_model = grid.best_estimator_
    overall_bacc = grid.best_score_
    
    corr_projections_by_frame = []
    for f in range(n_frames):
        corr_projections_by_frame.append(best_model.decision_function(X_correct_all_frames[f]))
        
    inc_projections_by_frame = []
    if X_incorrect_all_frames is not None and len(X_incorrect_all_frames[0]) > 0:
        for f in range(n_frames):
            inc_projections_by_frame.append(best_model.decision_function(X_incorrect_all_frames[f]))
            
    clf_step = best_model.named_steps['clf']
    model_info = {
        'weights': clf_step.coef_,
        'best_params': grid.best_params_,
        'overall_bacc': overall_bacc,
        'trained_frame_idx': train_frame_idx
    }
    
    return corr_projections_by_frame, inc_projections_by_frame, model_info

def process_session(one, session_id, significant_pickles, stim_region_name='VISp', choice_region_name='MOs', use_nested_cv=False, aggregate_parent=False, only_congruent=False):
    sig_stim = significant_pickles.get(session_id, {}).get('stim', [])
    sig_choice = significant_pickles.get(session_id, {}).get('choice', [])
    
    if stim_region_name not in sig_stim or choice_region_name not in sig_choice:
        logger.info(f"Skipping {session_id} - regions not significant.")
        return None
        
    ssl = SessionLoader(one, session_id)
    ssl.load_trials(collection="alf")
    trials = ssl.trials.copy()
    trials = compute_congruency(trials)
    
    _, behavioral_mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=False)
    
    stim_mask = behavioral_mask & (trials['signed_contrast'] != 0)
    choice_mask = behavioral_mask
    
    if only_congruent:
        stim_mask = stim_mask & trials['is_congruent']
        choice_mask = choice_mask & trials['is_congruent']
    
    correct_mask = trials["feedbackType"] == 1
    incorrect_mask = trials["feedbackType"] == -1
    
    stim_correct_idx = np.where(stim_mask & correct_mask)[0]
    stim_incorrect_idx = np.where(stim_mask & incorrect_mask)[0]
    
    choice_correct_idx = np.where(choice_mask & correct_mask)[0]
    choice_incorrect_idx = np.where(choice_mask & incorrect_mask)[0]
    
    if len(stim_correct_idx) < 5 or len(choice_correct_idx) < 5:
        logger.warning(f"Skipping {session_id} - not enough correct trials for 5-fold CV.")
        return None
        
    logger.info(f"Loading temporal widefield stim data for {stim_region_name}...")
    stim_data_list, stim_names = load_temporal_widefield_epoch(one, session_id, trials, config["hemisphere"], "stim", [stim_region_name], aggregate_parent=aggregate_parent)
    
    logger.info(f"Loading temporal widefield choice data for {choice_region_name}...")
    choice_data_list, choice_names = load_temporal_widefield_epoch(one, session_id, trials, config["hemisphere"], "choice", [choice_region_name], aggregate_parent=aggregate_parent)
    
    if len(stim_data_list) == 0 or len(choice_data_list) == 0:
        logger.warning(f"Skipping {session_id} - could not extract regions.")
        return None
        
    stim_data = stim_data_list[0]
    choice_data = choice_data_list[0]
    
    labels = trials["stim_side_computed"].values
    choice_labels = trials["choice"].values
    
    X_stim_correct = stim_data[:, stim_correct_idx, :]
    y_stim_correct = labels[stim_correct_idx]
    X_stim_incorrect = stim_data[:, stim_incorrect_idx, :] if len(stim_incorrect_idx) > 0 else None
    
    X_choice_correct = choice_data[:, choice_correct_idx, :]
    y_choice_correct = choice_labels[choice_correct_idx]
    X_choice_incorrect = choice_data[:, choice_incorrect_idx, :] if len(choice_incorrect_idx) > 0 else None
    
    project_fn = train_and_project_temporal_nested if use_nested_cv else train_and_project_temporal_single
    
    logger.info("Training Temporal Stimulus SVM (Frame 1)...")
    stim_svm_corr, stim_svm_inc, stim_svm_info = project_fn(X_stim_correct, y_stim_correct, X_stim_incorrect, train_frame_idx=1, classifier_name='svc')
    logger.info("Training Temporal Stimulus Ridge (Frame 1)...")
    stim_ridge_corr, stim_ridge_inc, stim_ridge_info = project_fn(X_stim_correct, y_stim_correct, X_stim_incorrect, train_frame_idx=1, classifier_name='ridge')
    
    logger.info("Training Temporal Choice SVM (Last Frame)...")
    choice_svm_corr, choice_svm_inc, choice_svm_info = project_fn(X_choice_correct, y_choice_correct, X_choice_incorrect, train_frame_idx=4, classifier_name='svc')
    logger.info("Training Temporal Choice Ridge (Last Frame)...")
    choice_ridge_corr, choice_ridge_inc, choice_ridge_info = project_fn(X_choice_correct, y_choice_correct, X_choice_incorrect, train_frame_idx=4, classifier_name='ridge')
    
    results_df = []
    
    for i, orig_idx in enumerate(stim_correct_idx):
        row = trials.iloc[orig_idx].to_dict()
        row['original_trial_index'] = orig_idx
        row['model_epoch'] = 'stim'
        row['is_correct_trial'] = True
        for f in range(5):
            row[f'svm_projection_f{f}'] = stim_svm_corr[f][i]
            row[f'ridge_projection_f{f}'] = stim_ridge_corr[f][i]
        results_df.append(row)
        
    for i, orig_idx in enumerate(stim_incorrect_idx):
        row = trials.iloc[orig_idx].to_dict()
        row['original_trial_index'] = orig_idx
        row['model_epoch'] = 'stim'
        row['is_correct_trial'] = False
        for f in range(5):
            row[f'svm_projection_f{f}'] = stim_svm_inc[f][i] if stim_svm_inc else np.nan
            row[f'ridge_projection_f{f}'] = stim_ridge_inc[f][i] if stim_ridge_inc else np.nan
        results_df.append(row)
        
    for i, orig_idx in enumerate(choice_correct_idx):
        row = trials.iloc[orig_idx].to_dict()
        row['original_trial_index'] = orig_idx
        row['model_epoch'] = 'choice'
        row['is_correct_trial'] = True
        for f in range(5):
            row[f'svm_projection_f{f}'] = choice_svm_corr[f][i]
            row[f'ridge_projection_f{f}'] = choice_ridge_corr[f][i]
        results_df.append(row)
        
    for i, orig_idx in enumerate(choice_incorrect_idx):
        row = trials.iloc[orig_idx].to_dict()
        row['original_trial_index'] = orig_idx
        row['model_epoch'] = 'choice'
        row['is_correct_trial'] = False
        for f in range(5):
            row[f'svm_projection_f{f}'] = choice_svm_inc[f][i] if choice_svm_inc else np.nan
            row[f'ridge_projection_f{f}'] = choice_ridge_inc[f][i] if choice_ridge_inc else np.nan
        results_df.append(row)
        
    results_df = pd.DataFrame(results_df)
    
    model_metadata = {
        'stim_svm_info': stim_svm_info,
        'stim_ridge_info': stim_ridge_info,
        'choice_svm_info': choice_svm_info,
        'choice_ridge_info': choice_ridge_info
    }
    
    return results_df, model_metadata

if __name__ == "__main__":
    one = ONE(mode="local")
    
    significant_pkl_path = Path("data/processed/significant_stims_choice.pkl")
    if significant_pkl_path.exists():
        with open(significant_pkl_path, "rb") as f:
            significant_pickles = pkl.load(f)
    else:
        logger.error(f"Could not find {significant_pkl_path}. Please check the path.")
        exit(1)

    USE_NESTED_CV = True
    AGGREGATE_PARENT = False
    ONLY_CONGRUENT = True

    for eid, regions in tqdm(significant_pickles.items()):
        if 'VISp' in regions.get('stim', []) and 'MOs' in regions.get('choice', []):
            logger.info(f"Running decoding on session: {eid}")

            try:
                results = process_session(
                    one, eid, significant_pickles, 
                    stim_region_name='VISp', choice_region_name='MOs', 
                    use_nested_cv=USE_NESTED_CV, aggregate_parent=AGGREGATE_PARENT, only_congruent=ONLY_CONGRUENT
                )
    
                if results is not None:
                    df, meta = results
                    out_dir = Path("data/generated/crosstime_decoders")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Determine suffix based on flags
                    suffix = "_temporal"
                    if ONLY_CONGRUENT:
                        suffix += "_congruent"
                    if not AGGREGATE_PARENT:
                        suffix += "_rawregions"
                    
                    df.to_parquet(out_dir / f"{eid}{suffix}_projections.pqt")
                    with open(out_dir / f"{eid}{suffix}_model_metadata.pkl", "wb") as f:
                        pkl.dump(meta, f)
                        
                    logger.info(f"Successfully saved test results to {out_dir}")
            except Exception as e:
                logger.error(e)
