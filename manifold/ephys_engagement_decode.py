import tempfile
import pickle
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

from one.api import ONE
from scipy.special import logit, softmax
import os
from os.path import join
import pickle as pkl
from decoding.functions.utils import check_config_decoding
from brainwidemap.bwm_loading import bwm_query, bwm_units, load_trials_and_mask, merge_probes
from decoding.fit_data import fit_session_ephys
import concurrent.futures

config = check_config_decoding()
MY_REGIONS = config["stim_prior_regions"]
MIN_NEURONS = config["min_units"]

# load animals, load engagement signal from dict, decode from region in animal that passes threshold.
# what about null distributions? (idk)


def fit_engagement(session_id, output_dir, engagement_signal, bwm_df):

    # i can load trials as normal

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
        # mode="local",
    )
    # try to make it local
    session_data = bwm_df[bwm_df["eid"] == session_id]
    pids = session_data["pid"].tolist()
    probes = session_data["probe_name"].tolist()
    # pids, probes = one.eid2pid(session_id)

    trials, mask = load_trials_and_mask(
        one,
        session_id,
        exclude_nochoice=False,
        exclude_unbiased=False,  # should include no-choice trials
    )
    trials["engagement"] = engagement_signal
    # trials = trials[mask]
    subject = bwm_df[bwm_df.eid == session_id].subject.iloc[0]
    results_dir = join(output_dir, subject, session_id)
    os.makedirs(results_dir, exist_ok=True)

    pseduosessions = np.arange(1, 101)
    pseduosessions_argument = np.concat([[-1], pseduosessions])
    try:
        results_fit_session = fit_session_ephys(
            one=one,
            session_id=session_id,
            subject=subject,
            pids=pids,
            probe_names=probes,
            output_dir=output_dir,
            model="oracle",
            pseudo_ids=pseduosessions_argument,
            align_event="stimOn_times",
            time_window=(-0.6, -0.1),
            n_runs=10,  # reduce this maybe : or change this based on pseudoids
            trials_df=trials,
            target="engagement",
            stage_only=True,
        )
    except Exception as e:
        _log = "Something wrong -- Skipping session"
        print("\t" + _log)
        print(_log, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(40 * "-")


if __name__ == "__main__":

    # config = check_config()
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    print("Querying BWM Units...")

    # only animals that pass the prior check
    # but go through all regions that those animals have
    # idk, this should be faster.
    units_df = bwm_units(one)
    relevant_pids = units_df[units_df["Beryl"].isin(MY_REGIONS)]["pid"].unique()

    bwm_df = bwm_query(one)
    runonalleids = bwm_df["eid"].unique()
    # change subset df: use all valid eids
    subset_df = bwm_df[bwm_df["pid"].isin(relevant_pids)]
    list_of_eids = subset_df["eid"].unique()

    # task_list = [(row["pid"], row["eid"]) for _, row in subset_df.iterrows()]

    print(config["engagement_dir"])
    print(config["output_dir"])

    engagement_dir = config["engagement_dir"]
    output_dir = config["output_dir"]

    with open(f"{engagement_dir}/all_eids_engagement.pkl", "rb") as f:
        engagement_pickle = pkl.load(f)

    def process_eid(eid):
        engagement_signal = engagement_pickle[eid]

        fit_engagement(
            session_id=eid,
            output_dir=output_dir,
            engagement_signal=engagement_signal,
            bwm_df=bwm_df,
        )

    # run a single one
    process_eid(list_of_eids[0])

    multiprocess = False
    if multiprocess:
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:

            futures = {
                executor.submit(process_eid, eid): eid for eid in runonalleids
            }  # NOTE: check which list we are passing

            for future in concurrent.futures.as_completed(futures):
                eid = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Session {eid} generated an exception: {exc}")
