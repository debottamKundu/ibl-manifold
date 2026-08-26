# a. load action kernel, pseudosessions and run a proper decoding to see which animals and sessions have above chance decoding
# b. then use those animals trained data, and look at how the strength of the prior influences everything.

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
import pickle as pkl
from glob import glob
from manifold.widefield_ppi import return_labels
from prior_localization.fit_data import fit_session_widefield
from pathlib import Path
from prior_localization.functions.utils import check_config


def run_fit(session_id, subject_name, n_pseudo=200):
    all_pseudo = list(range(n_pseudo))
    # Select relevant pseudo sessions for this job
    pseudo_ids = all_pseudo
    pseudo_ids = list(np.array(pseudo_ids) + 1)
    pseudo_ids = [-1] + pseudo_ids  # -1 is the true session
    x = fit_session_widefield(
        one=one,
        session_id=session_id,
        subject=subject_name,
        output_dir=Path("./data/generated/prior_sig"),
        pseudo_ids=pseudo_ids,
        hemisphere=("left", "right"),
        target="prior",
        align_event="stimOn_times",
        frame_window=(-2, -2),
        model="actKernel",
        n_runs=1,
    )
    return x

def combine_results_into_frame():
    filenames = glob('./data/generated/prior_sig/**/*_both*.pkl')
    resultsdf = []
    failedloads = 0
    indexers = ["subject", "eid", "region", "N_units"]
    for fname in filenames:
        try:
            with open(fname,'rb') as f:
                results = pkl.load(f)
            if results['fit'] is None:
                continue
            for iteration in range(len(results['fit'])):
                tmpdict = {**{x: results[x] for x in indexers},
                            "fold": -1,
                            "pseudo_id": results["fit"][iteration]["pseudo_id"],
                            "run_id": results["fit"][iteration]["run_id"] + 1,
                            "score_test": results["fit"][iteration]["scores_test_full"],
                            "n_trials": sum(results['fit'][iteration]['mask'][0]),
                    }
                resultsdf.append(tmpdict)
        except Exception as e:
            print(e)
            failedloads +=1
    resultsdf = pd.DataFrame(resultsdf)
    resultsdf.to_parquet('./data/generated/prior_sig/resultsdf.pqt')
    print(f'Failed loads:{failedloads}')


if __name__ == "__main__":
    one = ONE(mode="local")

    significant_pkl_path = Path("./data/processed/significant_stims_choice.pkl")

    with open(significant_pkl_path, "rb") as f:
        significant_pickles = pkl.load(f)

    # only take this eids
    sessionids = list(significant_pickles.keys())

    for session_id in sessionids:
        subject = one.get_details(session_id)["subject"]  # type: ignore
        target_folder = Path(f"./data/generated/prior_sig/{subject}/{session_id}/")
        if target_folder.exists():
            print(f'{session_id} exists')
            continue

        try:
            _ = run_fit(session_id, subject)
        except Exception as e:
            print(e)
        break
