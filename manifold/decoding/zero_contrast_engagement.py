import tempfile
import pickle
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
from brainbox.io.one import SessionLoader
from prior_localization.functions.utils import compute_mask
from one.api import ONE
from scipy.special import logit, softmax
import os
from os.path import join
import pickle as pkl
from decoding.functions.utils import check_config_decoding
from brainwidemap.bwm_loading import bwm_query, bwm_units, load_trials_and_mask, merge_probes
from decoding.fit_data_feedback import (
    fit_session_ephys,
)  # NOTE: maybe this confusion is not the best idea.
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler

config = check_config_decoding()


def engagement_zero_df(session_id, bwm_df, engagement_signal):

    one = ONE(
        # base_url="https://openalyx.internationalbrainlab.org",
        # password="international",
        # silent=True,
        # username="intbrainlab",
        mode="local",
    )

    session_data = bwm_df[bwm_df["eid"] == session_id]
    subject_name = session_data["subject"].unique()[0]

    # NOTE:widefield signals, don't use this.
    trials, mask = load_trials_and_mask(
        one,
        session_id,
        exclude_nochoice=False,
        exclude_unbiased=True,
    )
    trials["block_id"] = (trials["probabilityLeft"] != trials["probabilityLeft"].shift(1)).cumsum()
    trials["trial_within_block"] = trials.groupby("block_id").cumcount() + 1
    trials["engagement"] = engagement_signal
    out = np.nan_to_num(trials.contrastLeft) - np.nan_to_num(
        trials.contrastRight
    )  # this should keep 0 contrast
    trials["signcont"] = out
    trials["rt"] = trials["firstMovement_times"] - trials["stimOn_times"]
    trials = trials[mask]
    trials_keep = trials[trials["signcont"] == 0][
        [
            "engagement",
            "rt",
            "probabilityLeft",
            "feedbackType",
            "signcont",
            "choice",
            "trial_within_block",
        ]
    ]
    trials_keep["eid"] = session_id
    trials_keep["subject"] = subject_name

    return trials_keep


if __name__ == "__main__":

    # config = check_config()
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    bwm_df = bwm_query(one)
    runonalleids = bwm_df["eid"].unique()

    engagement_dir = config["engagement_dir"]
    # output_dir = config["output_dir"] + "_different_time_interval"

    with open(f"{engagement_dir}/all_eids_engagement.pkl", "rb") as f:
        engagement_pickle = pkl.load(f)

    def process_eid(eid):
        try:
            engagement_signal = np.asarray(engagement_pickle[eid])
            scalar = MinMaxScaler()
            engagement_signal = scalar.fit_transform(engagement_signal.reshape(-1, 1)).flatten()

            df = engagement_zero_df(
                session_id=eid,
                engagement_signal=engagement_signal,
                bwm_df=bwm_df,
            )
            return df
        except Exception as e:
            print(e)
            return pd.DataFrame()

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_eid, runonalleids))

    df = pd.concat(results)

    df.to_parquet("./data/generated/engagement_zero_contrast_scaled.pqt")
