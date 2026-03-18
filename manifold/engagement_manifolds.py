### new manifold structure
### what we need to do is problly collapse the left-right stimulus side stuff and just look at congruence-incongruence

import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns
import pickle as pkl
import manifold.reaction_time as mrx
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
from ibl_info.utils import check_config
from scipy.ndimage import convolve1d
import traceback
from scipy.stats import zscore
from ibl_info.pseudosession import fit_eid

config = check_config()
MY_REGIONS = config["stim_prior_regions"]
MIN_NEURONS = config["min_units"]
BIN_SIZE = 0.01  # 10ms
STRIDE = 0.001  # 1ms
USE_SLIDING_WINDOW = config["use_sliding_window"]
MIN_TRIALS = 1  # Minimum trials per condition to include session


# use reaction times rather than firstMovement times
# NOTE: change this on the server
with open("../data/processed/glm-hmm-output/glm_ouptut_all_animals.pkl", "rb") as f:
    trial_files = pkl.load(f)
with open("../data/processed/glm-hmm-output/action_kernel_trials.pkl", "rb") as f:
    action_kernel_files = pkl.load(f)

EPOCHS = {
    "Quiescent": {
        "align": "stimOn_times",
        "offset": -0.1,  # Align to -0.1s before Stim
        "t_pre": 0.5,
        "t_post": 0.0,
    }
}


def action_kernel_and_hmm_state(eid, congruent_split=True, include_history=False):
    """
    Generates masks splitting trials by GLM-HMM Engagement, Prior, Congruence, and Accuracy.
    """

    _, _, trials = mrx.load_trials_and_plot(trial_files, action_kernel_files, eid)

    expects_L = trials["akernel"] > 0.5
    expects_R = trials["akernel"] < 0.5

    has_contrast_L = ~np.isnan(trials["contrastLeft"])
    has_contrast_R = ~np.isnan(trials["contrastRight"])

    is_correct = trials["feedbackType"] == 1
    is_error = trials["feedbackType"] == -1

    is_engaged = trials["engagement_state"] == 1.0
    is_disengaged = trials["engagement_state"] == 0.0

    expected_side = np.where(trials["akernel"] > 0.5, -1, 1)
    stimulus_side = np.where(trials["contrastRight"].notna(), 1, -1)

    congruent = expected_side == stimulus_side
    incongruent = expected_side != stimulus_side

    prev_outcome = trials["feedbackType"].shift(1)
    prev_was_correct = prev_outcome == 1
    prev_was_error = prev_outcome == -1

    masks = {}

    if congruent_split == False:
        states = [("Eng", is_engaged), ("Dis", is_disengaged)]

        for state_name, state_mask in states:

            masks[f"{state_name}_L_Cong_Corr"] = (
                has_contrast_L & expects_L & is_correct & state_mask
            )
            masks[f"{state_name}_L_Cong_Err"] = has_contrast_L & expects_L & is_error & state_mask
            masks[f"{state_name}_R_Incong_Corr"] = (
                has_contrast_R & expects_L & is_correct & state_mask
            )
            masks[f"{state_name}_R_Incong_Err"] = (
                has_contrast_R & expects_L & is_error & state_mask
            )

            masks[f"{state_name}_R_Cong_Corr"] = (
                has_contrast_R & expects_R & is_correct & state_mask
            )
            masks[f"{state_name}_R_Cong_Err"] = has_contrast_R & expects_R & is_error & state_mask
            masks[f"{state_name}_L_Incong_Corr"] = (
                has_contrast_L & expects_R & is_correct & state_mask
            )
            masks[f"{state_name}_L_Incong_Err"] = (
                has_contrast_L & expects_R & is_error & state_mask
            )
    else:

        masks = {}
        meta_states = []
        if include_history:
            meta_states = [
                ("Eng", is_engaged),
                ("Dis_Hit", is_disengaged & prev_was_correct),
                ("Dis_Miss", is_disengaged & prev_was_error),
            ]
        else:
            meta_states = [("Eng", is_engaged), ("Dis", is_disengaged)]

        for s_name, s_mask in meta_states:
            for c_name, c_mask in [("Cong", congruent), ("Incong", incongruent)]:
                for a_name, a_mask in [("Corr", is_correct), ("Err", is_error)]:
                    label = f"{s_name}_{c_name}_{a_name}"
                    masks[label] = s_mask & c_mask & a_mask

    return masks, list(masks.keys())


def process_single_session(
    pid,
    eid,
    requested_regions,
    epochs_config,
    use_slide,
    win_size,
    stride,
    bin_simple,
):

    one_local = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    br_local = BrainRegions()
    session_results = {}

    try:
        spikes, clusters = load_good_units(one_local, pid)
        trials, trial_mask = load_trials_and_mask(
            one_local, eid, exclude_unbiased=True, exclude_nochoice=True
        )
        trials = {k: v[trial_mask] for k, v in trials.items()}

        all_spike_ids = clusters["cluster_id"][spikes["clusters"]]

        masks, cond_names = action_kernel_and_hmm_state(eid, congruent_split=True)
        COND_NAMES = cond_names

        for cond in COND_NAMES:
            if np.sum(masks[cond]) < MIN_TRIALS:
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

            for epoch_name, params in epochs_config.items():
                epoch_stack = []
                offset = params.get("offset", 0.0)

                for cond in COND_NAMES:
                    base_times = trials[params["align"]][masks[cond]].values
                    align_times = base_times + offset

                    if use_slide:
                        binned, _ = bin_spikes2D(
                            region_spike_times,
                            region_spike_ids,
                            target_ids,
                            align_times,
                            params["t_pre"],
                            params["t_post"],
                            stride,
                        )
                        w_points = int(win_size / stride)
                        kernel = np.ones(w_points) / w_points
                        smoothed = convolve1d(binned, kernel, axis=-1, mode="nearest")
                        psth = np.mean(smoothed, axis=0)
                    else:
                        binned, _ = bin_spikes2D(
                            region_spike_times,
                            region_spike_ids,
                            target_ids,
                            align_times,
                            params["t_pre"],
                            params["t_post"],
                            bin_simple,
                        )
                        psth = np.mean(binned, axis=0)

                    epoch_stack.append(psth)

                # Stack: (NeuronsxTime * 8_Conditions)
                session_results[region][epoch_name] = np.hstack(epoch_stack)

        return session_results

    except Exception as e:
        print(f"Error in {eid}: {e}")
        return None


def run_parallel(task_list):

    MAX_WORKERS = 8

    print(f"Found {len(task_list)} sessions. Starting extraction with {MAX_WORKERS} cores...")
    t0 = time.time()

    accumulated_data = {reg: {ep: [] for ep in "Quiescent"} for reg in MY_REGIONS}

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_session,
                pid,
                eid,
                MY_REGIONS,
                EPOCHS,
                USE_SLIDING_WINDOW,
                BIN_SIZE,
                STRIDE,
                BIN_SIZE,
            ): pid
            for (pid, eid) in task_list
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                for region, epoch_dict in result.items():
                    for epoch_name, matrix in epoch_dict.items():
                        # get all animals
                        accumulated_data[region][epoch_name].append(matrix)
            print(f"Progress: {i+1}/{len(task_list)}", end="\r")

    print(f"\nExtraction complete in {time.time() - t0:.2f} seconds.")

    save_path = f"./data/generated/bwm_data_engagement_ak.pkl"

    print(f"\nSaving data to {save_path}...")
    with open(save_path, "wb") as f:
        pkl.dump(accumulated_data, f)
    print("Save complete.")


if __name__ == "__main__":

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    print("Querying BWM Units...")

    units_df = bwm_units(one)
    relevant_pids = units_df[units_df["Beryl"].isin(MY_REGIONS)]["pid"].unique()

    bwm_df = bwm_query(one)
    subset_df = bwm_df[bwm_df["pid"].isin(relevant_pids)]

    task_list = [(row["pid"], row["eid"]) for _, row in subset_df.iterrows()]

    list_of_eids = subset_df["eid"].unique()

    run_parallel(task_list)
