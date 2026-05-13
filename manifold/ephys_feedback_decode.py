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
from decoding.fit_data_feedback import (
    fit_session_ephys_feedback,
)  # NOTE: maybe this confusion is not the best idea.
import concurrent.futures

config = check_config_decoding()
MY_REGIONS = config["prior_regions"]
MIN_NEURONS = config["min_units"]

# reduce locations, only prior locations
# change code, what this should do is
# get incongruent trials and decode feedback, if possible. that is all.


def decode_feedback(session_id, output_dir, bwm_df, n_pseudosession=200):

    # i can load trials as normal

    one = ONE(
        # base_url="https://openalyx.internationalbrainlab.org",
        # password="international",
        # silent=True,
        # username="intbrainlab",
        mode="local",
    )
    # try to make it local
    session_data = bwm_df[bwm_df["eid"] == session_id]
    pids = session_data["pid"].tolist()
    probes = session_data["probe_name"].tolist()

    trials, mask = load_trials_and_mask(
        one,
        session_id,
        exclude_nochoice=False,
        exclude_unbiased=True,  # should include no-choice trials
        min_rt=None,
        max_rt=None,
    )  # we only care about qs period anyways, and there is always feedback.

    # trials["engagement"] = engagement_signal
    # trials = trials[mask]
    subject = bwm_df[bwm_df.eid == session_id].subject.iloc[0]
    results_dir = join(output_dir, subject, session_id)
    print(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    pseduosessions = np.arange(1, n_pseudosession)
    pseduosessions_argument = np.concat([[-1], pseduosessions])
    try:
        results_fit_session = fit_session_ephys_feedback(
            one=one,
            session_id=session_id,
            subject=subject,
            pids=pids,
            probe_names=probes,
            output_dir=output_dir,
            model="actKernel",
            pseudo_ids=pseduosessions_argument,
            behavior_path="results_behavioral_zeta",
            align_event="stimOn_times",
            time_window=(-0.6, -0.1),
            n_runs=2,  # reduce this maybe : or change this based on pseudoids
            trials_df=trials,
            target="feedback",
            stage_only=False,
            incongruent_only=False,
            balanced_weighting=True,
            trials_mask=mask,
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

    units_df = bwm_units(one)
    # flattened_regions = [item for sublist in MY_REGIONS for item in sublist]
    # relevant_pids = units_df[units_df["Beryl"].isin(flattened_regions)]["pid"].unique()

    bwm_df = bwm_query(one)
    # change subset df: use all valid eids
    # no relevant pids, run on everything
    # subset_df = bwm_df[bwm_df["pid"].isin(relevant_pids)]
    # list_of_eids = subset_df["eid"].unique()
    runonalleids = bwm_df["eid"].unique()

    print(config["output_dir_feedback"])
    output_dir = config["output_dir_feedback"]

    def process_eid(eid, n_pseudosessions=200):
        decode_feedback(
            session_id=eid,
            output_dir=output_dir,
            bwm_df=bwm_df,
            n_pseudosession=n_pseudosessions,
        )

    # run a single one
    process_eid(runonalleids[0], 2)

    multiprocess = False  # NOTE: switch to true
    if multiprocess:
        with concurrent.futures.ProcessPoolExecutor(max_workers=120) as executor:

            futures = {
                executor.submit(process_eid, eid): eid for eid in runonalleids
            }  # NOTE: check which list we are passing

            for future in concurrent.futures.as_completed(futures):
                eid = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Session {eid} generated an exception: {exc}")
