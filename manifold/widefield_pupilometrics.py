# just get all the widefield pupilometrics data on disk

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

def load_pupilometrics(one, eid):
    sess = SessionLoader(one, eid)
    sess.load_pupil()
    pupil_df = sess.pupil
    return pupil_df


if __name__ == "__main__":
    
    one = ONE()
    sessions_all = one.search(datasets="widefieldU.images.npy")
    sessions_all = np.asarray([str(s) for s in sessions_all])  # type: ignore

    pupil_dict = {}
    for idx in range(len(sessions_all)):
        try: 
            eid = sessions_all[idx]
            pupil_df = load_pupilometrics(one,eid)
            pupil_dict[eid] = pupil_df
        except Exception as e:
            print(f'Problem with {eid}')
            pupil_dict[eid] = pd.DataFrame()
    
    with open('./data/pupilometry_widefield.pkl','wb') as f:
        pkl.dump(pupil_df, f)
