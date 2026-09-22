import os
import logging
import traceback
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from tqdm import tqdm
import uuid
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score

from one.api import ONE
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

def load_temporal_widefield_epoch_prior(one, session_id, trials, hemisphere, regions, aggregate_parent=False):
    """
    Extract window [-2, 5] relative to stimOn_times.
    """
    align_times = trials.stimOn_times
    frame_window = [-2, 5] # 8 frames total
        
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

def train_and_project_prior(X_train_all, y_train_binary, y_train_cont, X_eval_all, train_frame_idx, classifier_name='logreg'):
    n_frames = X_train_all.shape[0]
    n_train = y_train_binary.shape[0]
    
    X_train_target = X_train_all[train_frame_idx]
    
    corr_projections_by_frame = [np.zeros(n_train) for _ in range(n_frames)]
    inc_logits_folds = [[] for _ in range(n_frames)]
    models = []
    
    if classifier_name == 'logreg':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_model = LogisticRegression(penalty='l2', max_iter=10000, random_state=42)
        param_grid = {'clf__C': np.logspace(-3, 2, 6)}
        scoring = 'balanced_accuracy'
        cv_split = skf.split(X_train_target, y_train_binary)
        y_train_target = y_train_binary
    elif classifier_name == 'ridge':
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        base_model = Ridge(random_state=42)
        param_grid = {'clf__alpha': np.logspace(-3, 2, 6)}
        scoring = 'r2'
        cv_split = kf.split(X_train_target)
        y_train_target = y_train_cont
        
    oof_target_preds = np.zeros(n_train)
    
    for train_idx, test_idx in cv_split:
        X_train, y_train = X_train_target[train_idx], y_train_target[train_idx]
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', base_model)
        ])
        
        grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring=scoring)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        if classifier_name == 'logreg':
            # predict_proba returns (n_samples, n_classes). We want probability of class 1.
            class_1_idx = list(best_model.classes_).index(1)
            oof_target_preds[test_idx] = best_model.predict_proba(X_train_target[test_idx])[:, class_1_idx]
            for f in range(n_frames):
                corr_projections_by_frame[f][test_idx] = best_model.predict_proba(X_train_all[f][test_idx])[:, class_1_idx]
                if X_eval_all is not None and len(X_eval_all[0]) > 0:
                    inc_logits_folds[f].append(best_model.predict_proba(X_eval_all[f])[:, class_1_idx])
        else:
            oof_target_preds[test_idx] = best_model.predict(X_train_target[test_idx])
            for f in range(n_frames):
                corr_projections_by_frame[f][test_idx] = best_model.predict(X_train_all[f][test_idx])
                if X_eval_all is not None and len(X_eval_all[0]) > 0:
                    inc_logits_folds[f].append(best_model.predict(X_eval_all[f]))
                
        models.append({'weights': best_model.named_steps['clf'].coef_, 'best_params': grid.best_params_})
        
    if classifier_name == 'logreg':
        overall_score = balanced_accuracy_score(y_train_binary, (oof_target_preds > 0.5).astype(int)*2 - 1)
    else:
        overall_score = r2_score(y_train_cont, oof_target_preds)
    
    inc_projections_by_frame = []
    if X_eval_all is not None and len(X_eval_all[0]) > 0:
        for f in range(n_frames):
            inc_projections_by_frame.append(np.mean(inc_logits_folds[f], axis=0))
            
    model_info = {
        'models': models,
        'overall_score': overall_score,
        'trained_frame_idx': train_frame_idx
    }
    
    return corr_projections_by_frame, inc_projections_by_frame, model_info

def process_session(one, session_id, trials, region_name='VISp', aggregate_parent=False, only_congruent=False):
    trials = trials.copy()
    trials = compute_congruency(trials)
    
    _, behavioral_mask = load_trials_and_mask(one, str(session_id), exclude_nochoice=True, exclude_unbiased=False)
    
    stim_mask = behavioral_mask & (trials['signed_contrast'] != 0)
    
    valid_prior_mask = (trials['probabilityLeft'] != 0.5)
    
    correct_mask = trials["feedbackType"] == 1
    
    # Define training masks
    train_mask = stim_mask & correct_mask & valid_prior_mask
    if only_congruent:
        train_mask = train_mask & trials['is_congruent']
        
    # Define evaluation masks (everything in the epoch that wasn't used for training)
    eval_mask = stim_mask & ~train_mask
    
    train_idx = np.where(train_mask)[0]
    eval_idx = np.where(eval_mask)[0]
    
    if len(train_idx) < 5:
        logger.warning(f"Skipping {session_id} - not enough correct training trials for 5-fold CV.")
        return None
        
    logger.info(f"Loading temporal widefield data for {region_name}...")
    data_list, names = load_temporal_widefield_epoch_prior(one, str(session_id), trials, config["hemisphere"], [region_name], aggregate_parent=aggregate_parent)
    
    if len(data_list) == 0:
        logger.warning(f"Skipping {session_id} - could not extract regions.")
        return None
        
    data = data_list[0]
    
    prior_cont = trials["prior"].values
    prior_binary = np.sign(prior_cont - 0.5)
    
    X_train = data[:, train_idx, :]
    y_train_cont = prior_cont[train_idx]
    y_train_bin = prior_binary[train_idx]
    
    X_eval = data[:, eval_idx, :] if len(eval_idx) > 0 else None
    
    logger.info(f"Training Temporal Prior LogReg for {region_name} (Frame -2)...")
    logreg_train_proj, logreg_eval_proj, logreg_info = train_and_project_prior(X_train, y_train_bin, y_train_cont, X_eval, train_frame_idx=0, classifier_name='logreg')
    
    logger.info(f"Training Temporal Prior Ridge for {region_name} (Frame -2)...")
    ridge_train_proj, ridge_eval_proj, ridge_info = train_and_project_prior(X_train, y_train_bin, y_train_cont, X_eval, train_frame_idx=0, classifier_name='ridge')
    
    results_df = []
    
    for i, orig_idx in enumerate(train_idx):
        row = trials.iloc[orig_idx].to_dict()
        row['original_trial_index'] = orig_idx
        row['region'] = region_name
        row['is_correct_trial'] = (trials.iloc[orig_idx]["feedbackType"] == 1)
        row['used_for_training'] = True
        # Note: frame 0 -> -2, 1 -> -1, 2 -> 0, etc.
        for f in range(8):
            row[f'logreg_prob_f{f-2}'] = logreg_train_proj[f][i]
            row[f'ridge_pred_f{f-2}'] = ridge_train_proj[f][i]
        results_df.append(row)
        
    for i, orig_idx in enumerate(eval_idx):
        row = trials.iloc[orig_idx].to_dict()
        row['original_trial_index'] = orig_idx
        row['region'] = region_name
        row['is_correct_trial'] = (trials.iloc[orig_idx]["feedbackType"] == 1)
        row['used_for_training'] = False
        for f in range(8):
            row[f'logreg_prob_f{f-2}'] = logreg_eval_proj[f][i] if logreg_eval_proj else np.nan
            row[f'ridge_pred_f{f-2}'] = ridge_eval_proj[f][i] if ridge_eval_proj else np.nan
        results_df.append(row)
        
    results_df = pd.DataFrame(results_df)
    
    model_metadata = {
        f'{region_name}_logreg_info': logreg_info,
        f'{region_name}_ridge_info': ridge_info
    }
    
    return results_df, model_metadata


def run_single_session(zeta_dict, test_session):

    test_session = uuid.UUID(test_session)
    trials_df = zeta_dict[test_session]    
    logger.info(f"Running prior decoding test on session: {test_session}")
    
    all_results_df = []
    all_meta = {}
    
    for region in ['VISp', 'MOs']:
        try:
            results = process_session(one, test_session, trials_df, region_name=region, aggregate_parent=AGGREGATE_PARENT, only_congruent=ONLY_CONGRUENT)
            if results is not None:
                df, meta = results
                all_results_df.append(df)
                all_meta.update(meta)
        except Exception as e:
            logger.error(f"Failed on {region}: {e}")
            logger.error(traceback.print_exc())
            
    if all_results_df:
        final_df = pd.concat(all_results_df, ignore_index=True)
        out_dir = Path("data/generated/temporal_prior")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = "_prior_temporal"
        if ONLY_CONGRUENT:
            suffix += "_congruent"
        if not AGGREGATE_PARENT:
            suffix += "_rawregions"
            
        final_df.to_parquet(out_dir / f"{test_session}{suffix}_projections.pqt")
        with open(out_dir / f"{test_session}{suffix}_model_metadata.pkl", "wb") as f:
            pkl.dump(all_meta, f)
            
        logger.info(f"Successfully saved test results to {out_dir}")
        print(final_df[['original_trial_index', 'region', 'is_correct_trial', 'logreg_prob_f-2', 'ridge_pred_f-2']].head())


if __name__ == "__main__":
    one = ONE(mode="local")
    
    zeta_pkl_path = Path("data/generated/wifi/all_eids_dict_single_zeta_complete_wifi.pkl")
    if not zeta_pkl_path.exists():
        logger.error(f"Could not find {zeta_pkl_path}.")
        exit(1)

    eid_path = Path("data/generated/temporal_prior/significant_visp_eids.npy")
    eid_list = np.load(eid_path,allow_pickle=True)
        
    with open(zeta_pkl_path, "rb") as f:
        zeta_dict = pkl.load(f)
        
    AGGREGATE_PARENT = False
    ONLY_CONGRUENT = False

    for sessionid in tqdm(eid_list):
        run_single_session(zeta_dict=zeta_dict, test_session=sessionid)
