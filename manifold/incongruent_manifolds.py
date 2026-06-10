import concurrent.futures
import pickle as pkl
import time
from one.api import ONE
import pandas as pd
from tqdm import tqdm
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ibl_info.utils import (
    check_config,
    compute_animal_stats,
    get_action_kernel_congruence,
    get_trial_masks,
    get_trial_masks_detailed,
    action_kernel_and_previous_feedback,
)
from scipy.ndimage import convolve1d
import traceback
import os
from scipy.stats import zscore
from manifold.utils import get_trial_masks

# for all regions in the IBL with enough recordings
# get all incongruent trials
# get only correct/incorrect distinctions
# have one with 0 contrast trials
# have one without 0 contrast trials
# compute PSTHs.
# do PCA for visualization
# look at tests
# for null distributions, we need choices.
config = check_config()

MIN_NEURONS = 10
BIN_SIZE = 0.01
STRIDE = 0.001
USE_SLIDING_WINDOW = False

# only quiescent window
quiescent_window_params = {
    "align": "stimOn_times",
    "offset": -0.1,  # Align to -0.1s before Stim
    "t_pre": 0.5,
    "t_post": 0.0,
}

cond_names = ["Incongruent_correct", "Incongruent_incorrect"]


def process_single_session(
    pid,
    eid,
    requested_regions,
    epochs_config,
    bin_simple,
):
    """
    Loads one session, extracts spikes, and computes PETHs for 2 conditions.
    """
    one_local = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
        # mode="local",
    )
    br_local = BrainRegions()
    session_results = {}
    try:

        print(pid, eid)
        spikes, clusters = load_good_units(one_local, pid=pid, eid=eid)
        trials, trial_mask = load_trials_and_mask(
            one_local, eid, exclude_unbiased=True, exclude_nochoice=True
        )
        trials = trials[trial_mask]

        all_spike_ids = clusters["cluster_id"][spikes["clusters"]]

        congruency_masks, cond_names = get_trial_masks(trials)

        for cond in cond_names:
            if np.sum(congruency_masks[cond]) < 10:
                return None

        acronyms = br_local.id2acronym(clusters["atlas_id"], mapping="Beryl")
        for region in requested_regions:
            in_region = np.isin(acronyms, [region])
            if np.sum(in_region) < MIN_NEURONS:
                continue

            target_ids = clusters["cluster_id"][in_region]
            spike_mask = np.isin(all_spike_ids, target_ids)
            region_spike_times = spikes["times"][spike_mask]
            region_spike_ids = all_spike_ids[spike_mask]

            session_results[region] = {}

            epoch_stack = []
            offset = epochs_config.get("offset", 0.0)

            for cond in cond_names:
                base_times = trials[epochs_config["align"]][congruency_masks[cond]].values
                align_times = base_times + offset

                binned, _ = bin_spikes2D(
                    region_spike_times,
                    region_spike_ids,
                    target_ids,
                    align_times,
                    epochs_config["t_pre"],
                    epochs_config["t_post"],
                    bin_simple,
                )
                psth = np.mean(binned, axis=0)

                epoch_stack.append(psth)

            # Stack: (NeuronsxTime * Conditions)
            session_results[region] = np.hstack(epoch_stack)

        return session_results

    except Exception as e:
        print(f"Error in {eid}: {e}")
        return None


def run_parallel(task_list, regions, checkpoint_dir="./data/interim/session_checkpoints/"):

    MAX_WORKERS = 16
    print(f"Found {len(task_list)} sessions. Starting extraction with {MAX_WORKERS} cores...")
    t0 = time.time()

    os.makedirs(checkpoint_dir, exist_ok=True)

    aggregated_by_region = {region: {} for region in regions}

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_session,
                pid,
                eid,
                regions,
                quiescent_window_params,
                BIN_SIZE,
            ): (pid, eid)
            for (pid, eid) in task_list
        }

        for future in concurrent.futures.as_completed(futures):
            pid, eid = futures[future]

            try:
                session_results = future.result()

                if session_results is not None:

                    chkpt_filename = os.path.join(checkpoint_dir, f"session_{eid}_{pid}.pkl")
                    with open(chkpt_filename, "wb") as f:
                        pkl.dump(session_results, f)
                    for region, data in session_results.items():
                        aggregated_by_region[region][eid] = data

            except Exception as e:
                print(f"Worker crashed on {eid}: {e}")

        print("All sessions processed. Saving region pickles...")
        for region, region_dict in aggregated_by_region.items():

            if len(region_dict) > 0:
                filename = f"./data/generated/manifold/aggregated_{region}.pkl"
                with open(filename, "wb") as f:
                    pkl.dump(region_dict, f)
                print(f"Saved {region} ({len(region_dict)} sessions) to {filename}")
            else:
                print(f"Skipped {region} (0 valid sessions found)")

    print(f"Finished in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":

    regions_subset = np.asarray(
        [
            "MRN",
            "CA1",
            "DG",
            "CP",
            "LP",
            "SCm",
            "APN",
            "CA3",
            "PO",
            "PAG",
            "MOs",
            "VISp",
            "VPM",
            "VISa",
            "MOp",
            "ZI",
            "LSr",
            "IRN",
            "SUB",
            "CUL4 5",
            "IP",
            "PIR",
            "RSPv",
            "LGd",
            "SSp-bfd",
            "RSPd",
            "ACAd",
            "PRNr",
            "MV",
            "RT",
        ]
    )

    regions_all = np.asarray(
        [
            "CP",
            "MRN",
            "PO",
            "LP",
            "CA1",
            "SCm",
            "APN",
            "MOp",
            "VPM",
            "MOs",
            "CUL4 5",
            "LSr",
            "DG",
            "VISa",
            "SUB",
            "SIM",
            "CA3",
            "VISp",
            "PRNr",
            "IRN",
            "PAG",
            "MV",
            "CENT3",
            "ENTm",
            "IP",
            "CENT2",
            "PIR",
            "MD",
            "ENTl",
            "LD",
            "GRN",
            "LGd",
            "ANcr2",
            "RSPv",
            "SSp-bfd",
            "IC",
            "AON",
            "SSs",
            "ANcr1",
            "VPL",
            "RN",
            "RT",
            "PL",
            "PARN",
            "CS",
            "PRM",
            "ProS",
            "VISam",
            "ACAd",
            "PPN",
            "Eth",
            "SSp-m",
            "ACB",
            "MG",
            "SPIV",
            "ZI",
            "GPe",
            "SPVI",
            "DP",
            "DCO",
            "RSPd",
            "RSPagl",
            "SSp-tr",
            "PB",
            "COPY",
            "SUV",
            "TTd",
            "SI",
            "VISpm",
            "POST",
            "PoT",
            "BST",
            "ACAv",
            "BMA",
            "SCs",
            "LHA",
            "CEA",
            "LSv",
            "POL",
            "BLA",
            "SNr",
            "VM",
            "NTS",
        ]
    )

    # first we run for subset
    test = False
    multiprocess = True

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    print("Querying BWM Units...")

    units_df = bwm_units(one)
    relevant_pids = units_df[units_df["Beryl"].isin(regions_subset)]["pid"].unique()

    bwm_df = bwm_query(one)
    subset_df = bwm_df[bwm_df["pid"].isin(relevant_pids)]
    task_list = [(row["pid"], row["eid"]) for _, row in subset_df.iterrows()]

    list_of_eids = subset_df["eid"].unique()

    # run one
    if test:
        eid = "0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde"
        pid, probes = one.eid2pid(eid)
        session_results = process_single_session(
            pid=pid[0],
            eid=eid,
            requested_regions=regions_all,
            epochs_config=quiescent_window_params,
            bin_simple=BIN_SIZE,
        )
        if session_results is not None:
            print(len(session_results))  # type: ignore
        session_results = process_single_session(
            pid=pid[1],
            eid=eid,
            requested_regions=regions_all,
            epochs_config=quiescent_window_params,
            bin_simple=BIN_SIZE,
        )
        if session_results is not None:
            print(len(session_results))  # type: ignore

    if multiprocess:
        run_parallel(task_list, regions_subset)
