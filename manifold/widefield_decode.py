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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("decoding_pipeline_debug.log")],
)
logger = logging.getLogger(__name__)

BEHAVIOR_PATH = (
    "./results_behavioral_zeta/"  # NOTE: this is for choice
)
STIM_FRAME = 0  # onset frame
CHOICE_FRAME = 1  # movement onset frame
config = check_config_decoding()


def get_frame(epoch):
    if epoch == "stim":
        return STIM_FRAME
    elif epoch == "choice":
        return CHOICE_FRAME
    else:
        raise ValueError


def prepare_behavior(trials, session_id, mask, epoch, pseudosessions=200):
    logger.info(
        f"Preparing behavior targets for epoch: '{epoch}' with {pseudosessions} pseudosessions."
    )

    if epoch == "stim":
        true_targets = trials[mask]["stim_side"].values
        pseudo_targets = []
        for psession in range(pseudosessions):
            null_trials = nulldistributions.generate_null_distribution_session(
                trials, session_id, "john-doe", "notAK", BEHAVIOR_PATH
            )
            pseudo_targets.append(null_trials[mask]["stim_side"])
        pseudo_targets = np.array(pseudo_targets)
    elif epoch == "choice":
        true_targets = trials[mask]["choice"].values
        pseudo_targets = []
        for psession in range(pseudosessions):
            null_trials = nulldistributions.generate_null_distribution_session(
                trials, session_id, "john-doe", "actKernel", BEHAVIOR_PATH
            )
            pseudo_targets.append(null_trials[mask]["choice"])
        pseudo_targets = np.array(pseudo_targets)

    return true_targets, pseudo_targets


def decode_cv(X, Y, outer_cv_splits=5, inner_cv_splits=3, random_state=42):
    """
    X: ntrials x featues
    Y: ntrials
    """
    outer_cv = StratifiedKFold(n_splits=outer_cv_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(len(Y), dtype=float)
    param_grid = {
        "classifier__C": np.logspace(-2, 2, 5),
        "classifier__penalty": ["l2"],
    }
    outer_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        pipeline = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=1,
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred_probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        oof_predictions[test_idx] = y_pred_probs
        fold_score = balanced_accuracy_score(y_test, y_pred)

        outer_scores.append(fold_score)

    results = {
        "oof_predictions": oof_predictions,
        "outer_scores": outer_scores,
        "mean_score": np.mean(outer_scores),
        "std_score": np.std(outer_scores),
    }
    return results


def fit_target(neural_data, target, pseudo_target):
    true_results = decode_cv(neural_data, target)

    mean_scores_for_pseudosessions = []
    for psession in tqdm(range(pseudo_target.shape[0])):
        pseudo_results = decode_cv(neural_data, pseudo_target[psession, :])
        mean_scores_for_pseudosessions.append(pseudo_results["mean_score"])

    true_results["pseudoessions"] = mean_scores_for_pseudosessions
    return true_results


def run_single_animal(session_id, epoch="stim", n_pseudosessions=200, subepoch=None):
    logger.info(f"--- Starting session: {session_id} | Epoch: {epoch} ---")

    one = ONE(mode="local")
    ssl = SessionLoader(one, session_id)
    ssl.load_trials(collection="alf")
    trials = ssl.trials.copy()

    out = np.nan_to_num(trials.contrastLeft) - np.nan_to_num(trials.contrastRight)
    trials["signcont"] = out
    trials["stim_side"] = np.sign(trials["signcont"])

    logger.info(f"Loading mask for '{epoch}'...")
    if epoch == "stim":
        _, stimulus_behavioral_mask = load_trials_and_mask(
            one,
            session_id,
            exclude_nochoice=False,
            exclude_unbiased=False,
        )
        stimulus_non_zero_mask = trials["signcont"] != 0
        high_contrast_mask = np.abs(trials["signcont"]) == 1
        hhmask = stimulus_behavioral_mask & high_contrast_mask
        mask = stimulus_behavioral_mask & stimulus_non_zero_mask
    elif epoch == "choice":
        _, choice_behavioral_mask = load_trials_and_mask(
            one,
            session_id,
            exclude_nochoice=True,
            exclude_unbiased=False,
        )
        mask = choice_behavioral_mask
    else:
        logger.error(f"Invalid epoch provided: {epoch}")
        raise ValueError

    logger.info(f"Mask applied. {mask.sum()} valid trials remaining out of {len(mask)}.")

    target, pseudo_targets = prepare_behavior(
        trials, session_id, mask, epoch, pseudosessions=n_pseudosessions
    )

    logger.info("Loading widefield epoch data...")
    data, region_names = load_widefield_epoch(
        one, session_id, trials, config["hemisphere"], epoch=epoch, response_time=True
    ) # we can also use response time = False to default back to firstMovement times. 

    results = {}
    logger.info(f"Running decoding across {len(region_names)} regions...")

    for subepoch_shadow in ["choice-1", "choice-2"]: # stim-1, stim-2, choice-1, choice-2? 
        results_subepoch = {}
        if subepoch_shadow  == 'choice-1':
            continue
        print(subepoch_shadow)
        for region_data, region_name in tqdm(
            zip(data, region_names), desc="Regions", total=len(region_names)
        ):

            if subepoch_shadow == "choice-1":
                neural_data = region_data[0, mask, :]
            elif subepoch_shadow == "choice-2":
                neural_data = region_data[1, mask, :]
            else:
                raise ValueError

            results_subepoch[region_name[0]] = fit_target(neural_data, target, pseudo_targets)

            region_mean = results_subepoch[region_name[0]]["mean_score"]
            region_null = np.nanmedian(results_subepoch[region_name[0]]["pseudoessions"])
            logger.info(
                f"Region: {region_name[0]} | Mean Score: {region_mean:.4f} | Median Null: {region_null:.4f}"
            )
        results[subepoch_shadow] = results_subepoch

    logger.info(f"Successfully finished session: {session_id} for epoch: {epoch}")
    return results


def process_session_epoch(session, epoch, subepoch=None):
    try:
        #   print(session, subepoch)
        # fast execution, we treat change as null
        results = run_single_animal(session, epoch=epoch, subepoch=subepoch, n_pseudosessions=200)
        output_path = f"./data/generated/wifi/{session}_{subepoch}.pkl"

        with open(output_path, "wb") as f:
            pkl.dump(results, f)
            logger.info(f"Successfully saved results to {output_path}")
        return True

    except Exception as e:
        logger.error(
            f"Failed processing session {session} for epoch '{subepoch}'. Error: {e}",
            exc_info=True,
        )
        return False


if __name__ == "__main__":
    logger.info("Initializing ONE and searching for datasets...")
    one = ONE(mode="local")
    sessions_all = one.search(datasets="widefieldU.images.npy")
    sessions_all = np.asarray([str(s) for s in sessions_all])  # type: ignore

    # since we already ran the first onee
    # sessions_all = sessions_all[3:]

    logger.info(f"Found {len(sessions_all)} sessions with widefield data.")

    # for session in sessions_all:
    #     for epoch in ["stim", "choice"]:
    #         try:
    #             results = run_single_animal(session, epoch)
    #             output_path = f"./data/generated/wifi/{session}_{epoch}.pkl"
    #             with open(output_path, "wb") as f:
    #                 pkl.dump(results, f)
    #             logger.info(f"Saved results to {output_path}")
    #         except Exception as e:

    #             logger.error(
    #                 f"Failed processing session {session} for epoch '{epoch}'. Error: {e}",
    #                 exc_info=True,
    #             )
    #     break  # NOTE: remove this once the first one runs properly

    # subepochs = ["stim-both"]
    epoch = 'choice'
    tasks = list(itertools.product(sessions_all, epoch))

    total_tasks = len(tasks)
    successful_tasks = 0

    logger.info(f"Starting parallel processing for {total_tasks} tasks...")

    # run single to see if this ends?
    # ss, sube = tasks[0]
    # results = run_single_animal(ss, subepoch=sube, n_pseudosessions=1)
    for session, epoch in tasks:
        try:
            success = process_session_epoch(session, epoch)
            if success:
                successful_tasks += 1
        except Exception as e:
            logger.error(f"Task {(session, epoch)} generated an unexpected exception: {e}")

    logger.info(
        f"Processing complete! Successfully processed {successful_tasks}/{total_tasks} tasks."
    )
    multiprocess = True
    if multiprocess:
        with ProcessPoolExecutor(max_workers=5) as executor:
            future_to_task = {
                executor.submit(process_session_epoch, session, subepoch): (session, subepoch)
                for session, subepoch in tasks
            }

            for future in as_completed(future_to_task):
                session, epoch = future_to_task[future]
                try:

                    success = future.result()
                    if success:
                        successful_tasks += 1
                except Exception as exc:

                    logger.error(
                        f"Task {(session, epoch)} generated an unexpected exception: {exc}"
                    )

            logger.info(
                f"Processing complete! Successfully processed {successful_tasks}/{total_tasks} tasks."
            )
