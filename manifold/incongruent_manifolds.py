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
from manifold.decoding.functions import nulldistributions
from manifold.utils import get_trial_masks, get_trial_masks_engagement


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
BEHAVIOR_PATH = "./results_behavioral_zeta/"  # NOTE: this is for choice
# only quiescent window
quiescent_window_params = {
    "align": "stimOn_times",
    "offset": -0.1,  # Align to -0.1s before Stim
    "t_pre": 0.5,
    "t_post": 0.0,
}

cond_names = ["Incongruent_correct", "Incongruent_incorrect", "Congruent_correct", "Congruent_incorrect"]


def subsample_and_average_psth(binned_trials, target_n, n_iterations=100):
    """
    Subsamples trials to a target number and averages to create a stable PSTH.
    binned_trials shape: (n_trials, n_neurons * n_bins)
    """
    n_trials = binned_trials.shape[0]

    print(n_trials, target_n)
    if n_trials == target_n:
        return np.mean(binned_trials, axis=0)

    subsampled_psths = []
    for iter in range(n_iterations):
        idx = np.random.choice(n_trials, target_n, replace=False)
        subsampled_psths.append(np.mean(binned_trials[idx], axis=0))

    return np.mean(subsampled_psths, axis=0)


def generate_pseudosessions_incongruent_conditions(trials, session_id, n_pseudosessions=200):

    pseudo_masks = []
    for psession in range(n_pseudosessions):
        null_trials = nulldistributions.generate_null_distribution_session(
            trials, session_id, "john-doe", "actKernel", BEHAVIOR_PATH
        )
        masks, _ = get_trial_masks(null_trials)

        pseudo_masks.append(masks)
    return pseudo_masks


def process_single_session(
    pid, eid, requested_regions, epochs_config, bin_simple, pseudosession=False,n_pseudosessions=200, engagement=False, engagement_df=None
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
        if engagement:
            if engagement_df is None:
                raise ValueError
            engagement_df_eid = engagement_df[engagement_df['eid']==eid].reset_index(drop=True)
            trials = trials.merge(engagement_df_eid[['p_state1','p_state2','signed_contrast','rewarded']],left_index=True, right_index=True)
            assert np.all(trials['feedbackType']==trials['rewarded'])
        # this should work!

        trials = trials[trial_mask]

        all_spike_ids = clusters["cluster_id"][spikes["clusters"]]

        if engagement:
            condition_masks, cond_names = get_trial_masks_engagement(trials)
        else:
            condition_masks, cond_names = get_trial_masks(trials)

        for cond in cond_names:
            if np.sum(condition_masks[cond]) < 10:
                return None
            else:
                print(cond, np.sum(condition_masks[cond]))

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

            offset = epochs_config.get("offset", 0.0)

            if not pseudosession:

                binned_conditions = {}
                for cond in cond_names:
                    base_times = trials[epochs_config["align"]][condition_masks[cond]].values
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
                    binned_conditions[cond] = binned
                    # psth = np.mean(binned, axis=0)
                min_trials = min([binned.shape[0] for binned in binned_conditions.values()])
                epoch_stack = []
                for cond in cond_names:
                    psth = subsample_and_average_psth(
                        binned_conditions[cond], min_trials, n_iterations=10
                    )
                    epoch_stack.append(psth)
                # Stack: (NeuronsxTime * Conditions)
                session_results[region] = np.hstack(epoch_stack)
            else:
                pseudo_masks = generate_pseudosessions_incongruent_conditions(
                    trials, eid, n_pseudosessions=n_pseudosessions
                )
                pseudosession_epochs = []
                for idx in range(len(pseudo_masks)):
                    mask_prime = pseudo_masks[idx]
                    binned_conditions = {}
                    for cond in cond_names:
                        base_times = trials[epochs_config["align"]][mask_prime[cond].values].values
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
                        # psth = np.mean(binned, axis=0)
                        binned_conditions[cond] = binned
                    min_trials = min([binned.shape[0] for binned in binned_conditions.values()])
                    epoch_stack = []
                    for cond in cond_names:
                        binned_array = binned_conditions[cond]
                        if binned_array.shape[0] > min_trials:
                            rand_idx = np.random.choice(
                                binned_array.shape[0], min_trials, replace=False
                            )
                            psth = np.mean(binned_array[rand_idx], axis=0)
                        else:
                            psth = np.mean(binned_array, axis=0)

                        epoch_stack.append(psth)

                    pseudosession_epochs.append(np.hstack(epoch_stack))
                session_results[region] = pseudosession_epochs

        return session_results

    except Exception as e:
        print(f"Error in {eid}: {e}")
        return None


def run_parallel(
    task_list,
    regions,
    pseudosession=False,
    engagement_session=False,
    checkpoint_dir="./data/interim/session_checkpoints/",
):

    MAX_WORKERS = 32
    print(f"Found {len(task_list)} sessions. Starting extraction with {MAX_WORKERS} cores...")
    t0 = time.time()

    os.makedirs(checkpoint_dir, exist_ok=True)

    aggregated_by_region = {region: {} for region in regions}
    if engagement_session==False:
        engagement_df = None
    else:
        engagement_df = pd.read_parquet('./data/external/merged_behavioral_and_states.pqt')

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_session,
                pid,
                eid,
                regions,
                quiescent_window_params,
                BIN_SIZE,
                pseudosession,
                1000,
                engagement_session, 
                engagement_df,    
            ): (pid, eid)
            for (pid, eid) in task_list
        }

        for future in concurrent.futures.as_completed(futures):
            pid, eid = futures[future]

            try:
                session_results = future.result()

                if session_results is not None:
                    if not pseudosession:
                        chkpt_filename = os.path.join(checkpoint_dir, f"session_{eid}_{pid}.pkl")
                    else:
                        chkpt_filename = os.path.join(
                            checkpoint_dir, f"session_{eid}_{pid}_pseudosession.pkl"
                        )
                    with open(chkpt_filename, "wb") as f:
                        pkl.dump(session_results, f)
                    for region, data in session_results.items():
                        aggregated_by_region[region][eid] = data

            except Exception as e:
                print(f"Worker crashed on {eid}: {e}")

        print("All sessions processed. Saving region pickles...")
        for region, region_dict in aggregated_by_region.items():

            if len(region_dict) > 0:
                if not pseudosession:
                    filename = f"./data/generated/manifold/aggregated_{region}.pkl"
                else:
                    filename = f"./data/generated/manifold/aggregated_{region}_pseudosession.pkl"
                with open(filename, "wb") as f:
                    pkl.dump(region_dict, f)
                print(f"Saved {region} ({len(region_dict)} sessions) to {filename}")
            else:
                print(f"Skipped {region} (0 valid sessions found)")

    print(f"Finished in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":

    # regions_subset = np.asarray(
    #     [
    #         "MRN",
    #         "CA1",
    #         "DG",
    #         "CP",
    #         "LP",
    #         "SCm",
    #         "APN",
    #         "CA3",
    #         "PO",
    #         "PAG",
    #         "MOs",
    #         "VISp",
    #         "VPM",
    #         "VISa",
    #         "MOp",
    #         "ZI",
    #         "LSr",
    #         "IRN",
    #         "SUB",
    #         "CUL4 5",
    #         "IP",
    #         "PIR",
    #         "RSPv",
    #         "LGd",
    #         "SSp-bfd",
    #         "RSPd",
    #         "ACAd",
    #         "PRNr",
    #         "MV",
    #         "RT",
    #     ]
    # )

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

    # regions_difference = np.asarray(
    #     ['SIM',
    #     'CEA',
    #     'VISam',
    #     'BST',
    #     'ACAv',
    #     'VISpm',
    #     'MG',
    #     'LHA',
    #     'GPe',
    #     'DP',
    #     'PRM',
    #     'ACB',
    #     'SCs',
    #     'PoT',
    #     'SI',
    #     'SNr',
    #     'MD',
    #     'Eth',
    #     'ProS',
    #     'POST',
    #     'IC',
    #     'SSp-m',
    #     'SPVI',
    #     'PB',
    #     'LD',
    #     'POL',
    #     'GRN',
    #     'SSp-tr',
    #     'ANcr2',
    #     'CENT3',
    #     'ENTl',
    #     'NTS',
    #     'CENT2',
    #     'SSs',
    #     'BMA',
    #     'SUV',
    #     'ANcr1',
    #     'AON',
    #     'BLA',
    #     'VM',
    #     'LSv',
    #     'PPN',
    #     'RN',
    #     'VPL',
    #     'PARN',
    #     'ENTm',
    #     'DCO',
    #     'PL',
    #     'RSPagl',
    #     'SPIV',
    #     'COPY',
    #     'CS',
    #     'TTd'
    #     ]
    # )
    # regions_significant
    # regions_of_interest = np.asarray([
    #     "ACB",
    #     "IRN",
    #     "CA1",
    #     "SUV",
    #     "LSr",
    #     "DG",
    #     "PO",
    #     "CP",
    #     "APN",
    #     "SCm",
    #     "GRN",
    #     "SIM",
    #     "MG",
    #     "ACAd",
    #     "ZI",
    #     "VISa",
    #     "MRN",
    #     "VPL",
    #     "MD",
    #     "POL",
    #     "PARN",
    #     "SSp-bfd",
    #     "MV",
    #     "LP",
    #     "LGd",
    #     "RN",
    #     "SSs",
    #     "CA3",
    #     "PAG",
    #     "SI",
    #     "VPM",
    #     "PRNr",
    #     "PoT",
    #     "IP",
    #     "CENT3",
    #     "MOp",
    #     "LD",
    #     "RT",
    #     "PPN",
    # ])
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
    relevant_pids = units_df[units_df["Beryl"].isin(regions_all)]["pid"].unique()

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
            pseudosession=False,
        )
        if session_results is not None:
            print(len(session_results))  # type: ignore
            with open("./data/generated/manifold/test.pkl", "wb") as f:
                pkl.dump(session_results, f)

    if multiprocess:
        run_parallel(task_list, regions_all, pseudosession=False)
        
