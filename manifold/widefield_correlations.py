from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import logging
from one.api import ONE
from brainbox.io.one import SessionLoader
from sklearn.decomposition import PCA
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
    handlers=[logging.StreamHandler(), logging.FileHandler("correlation.log")],
)
logger = logging.getLogger(__name__)


config = check_config_decoding()


def compute_correlations_pca_dimensions(data, threshold=0.9):
    # trials x voxels
    n_trials, n_voxels = data.shape

    corr_matrix = np.corrcoef(data, rowvar=False)
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = np.nanmean(corr_matrix[upper_tri_indices])

    max_components = min(n_trials, n_voxels)
    pca = PCA(n_components=max_components)
    pca.fit(data)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_thresh = np.argmax(cumulative_variance >= threshold) + 1

    # print(n_components_thresh, mean_corr)
    return mean_corr, n_components_thresh


def run_single_animal(session_id):
    logger.info(f"--- Starting session: {session_id} ---")

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
    stimulus_mask = stimulus_behavioral_mask & stimulus_non_zero_mask

    _, choice_behavioral_mask = load_trials_and_mask(
        one,
        session_id,
        exclude_nochoice=True,
        exclude_unbiased=False,
    )

    logger.info("Loading widefield epoch data...")
    data_stimulus, region_names = load_widefield_epoch(
        one, session_id, trials, config["hemisphere"], epoch="stim"
    )
    data_choice, _ = load_widefield_epoch(
        one, session_id, trials, config["hemisphere"], epoch="choice"
    )

    results = {}
    logger.info(f"Running decoding across {len(region_names)} regions...")

    for idx in range(len(data_stimulus)):
        temp = {}
        region_data_stim = data_stimulus[idx]
        region_name = region_names[idx]

        neural_data_stim_frame_zero = region_data_stim[0, stimulus_mask, :]
        neural_data_stim_frame_one = region_data_stim[1, stimulus_mask, :]

        region_data_choice = data_choice[idx]
        neural_data_choice_frame_zero = region_data_choice[0, choice_behavioral_mask, :]
        neural_data_choice_frame_one = region_data_choice[1, choice_behavioral_mask, :]

        temp["stim_zero"] = np.asarray(
            compute_correlations_pca_dimensions(neural_data_stim_frame_zero)
        )
        temp["stim_one"] = np.asarray(
            compute_correlations_pca_dimensions(neural_data_stim_frame_one)
        )

        temp["choice_zero"] = np.asarray(
            compute_correlations_pca_dimensions(neural_data_choice_frame_zero)
        )
        temp["choice_one"] = np.asarray(
            compute_correlations_pca_dimensions(neural_data_choice_frame_one)
        )

        results[region_name[0]] = temp

    logger.info(f"Successfully finished session: {session_id}")
    return results


def process_session_epoch(session):
    try:
        #   print(session, subepoch)
        # fast execution, we treat change as null
        results = run_single_animal(session)
        output_path = f"./data/generated/wifi/{session}_correlations.pkl"

        with open(output_path, "wb") as f:
            pkl.dump(results, f)
            logger.info(f"Successfully saved results to {output_path}")
        return True

    except Exception as e:
        logger.error(
            f"Failed processing session {session}'. Error: {e}",
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

    # run single to see if this ends?
    # ss, sube = tasks[0]
    # results = run_single_animal(ss, subepoch=sube, n_pseudosessions=1)
    successful_tasks = 0
    for session in sessions_all:
        try:
            success = process_session_epoch(session)
            if success:
                successful_tasks += 1
        except Exception as e:
            logger.error(f"Task {(session)} generated an unexpected exception: {e}")
    logger.info(f"Total succesful tasks: {successful_tasks} / {len(sessions_all)}")

    logger.info("--FIN--")
