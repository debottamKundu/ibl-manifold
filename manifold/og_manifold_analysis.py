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
from scipy.stats import zscore
from ibl_info.pseudosession import fit_eid

config = check_config()
MY_REGIONS = config["stim_prior_regions"]
MIN_NEURONS = config["min_units"]
BIN_SIZE = 0.01  # 10ms
STRIDE = 0.001  # 1ms
USE_SLIDING_WINDOW = config["use_sliding_window"]
MIN_TRIALS = 1  # Minimum trials per condition to include session

EPOCHS = {
    "Quiescent": {
        "align": "stimOn_times",
        "offset": -0.1,  # Align to -0.1s before Stim
        "t_pre": 0.5,
        "t_post": 0.0,
    },
    "Stimulus": {"align": "stimOn_times", "offset": 0.0, "t_pre": 0.0, "t_post": 0.15},
    "Choice": {"align": "firstMovement_times", "offset": 0.0, "t_pre": 0.15, "t_post": 0.0},
}

COND_NAMES = [
    "L_Cong_Corr",
    "L_Cong_Err",
    "L_Incong_Corr",
    "L_Incong_Err",
    "R_Cong_Corr",
    "R_Cong_Err",
    "R_Incong_Corr",
    "R_Incong_Err",
]

# Base colors for the 4 types (Dark Blue, Light Blue, Dark Red, Light Red)
# We will use Solid Lines for Correct, Dashed for Error
BASE_COLORS = ["#00008B", "#6495ED", "#8B0000", "#FA8072"]


def process_single_session(
    pid,
    eid,
    requested_regions,
    epochs_config,
    use_slide,
    win_size,
    stride,
    bin_simple,
    difficulty=0,
):
    """
    Loads one session, extracts spikes, and computes PETHs for 8 conditions.
    """
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

        # Get masks for all 8 conditions
        if difficulty == 0:
            masks, cond_names = get_trial_masks(trials, simple=True)
        elif difficulty == 1:
            masks, cond_names = get_trial_masks(trials)
        elif difficulty == 2:
            masks, cond_names = action_kernel_and_previous_feedback(
                eid, trial_mask, only_corr=True
            )
        elif difficulty == 3:
            masks, cond_names = action_kernel_and_previous_feedback(
                eid, trial_mask, only_corr=False
            )
        elif difficulty == 4:
            masks, cond_names = action_kernel_and_previous_feedback(
                eid, trial_mask, only_corr=False, smoothed=True
            )
        elif difficulty == 6:
            masks, cond_names = get_action_kernel_congruence(eid, trial_mask)
        elif difficulty == 7:
            masks, cond_names = get_action_kernel_congruence(eid, trial_mask, only_corr=False)
        else:
            raise NotImplementedError
        COND_NAMES = cond_names
        # if simple_mask:
        #     COND_NAMES = ["Left", "Right"]

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


def plot_pcas_separate_decomposition(accumulated_data, region, conditions=8):

    print(f"\n--- Visualizing Region: {region} ---")

    epochs_ordered = ["Quiescent", "Stimulus", "Choice"]

    fig1 = plt.figure(figsize=(15, 5))
    fig1.suptitle(f"{region} | PCA Common Space", fontsize=16)

    for i, epoch_name in enumerate(epochs_ordered):
        session_matrices = accumulated_data[region][epoch_name]
        if not session_matrices:
            continue

        pop_matrix = np.vstack(session_matrices)

        pca = PCA(n_components=3)
        X_embedded = pca.fit_transform(pop_matrix.T)  # (Total_Time, 3)

        n_bins = int(X_embedded.shape[0] / conditions)  # Divided by conditions conditions
        trajs = X_embedded.reshape(conditions, n_bins, 3)

        ax = fig1.add_subplot(1, 3, i + 1, projection="3d")

        for c in range(conditions):
            color_idx = c // 2  # 0 or 1 -> 0; 2 or 3 -> 1, etc.
            is_err = (c % 2) != 0  # Even indices are Corr, Odd are Err

            style = "--" if is_err else "-"
            alpha = 0.6 if is_err else 1.0
            lw = 1.5 if is_err else 2.5

            ax.plot(
                trajs[c, :, 0],
                trajs[c, :, 1],
                trajs[c, :, 2],
                color=BASE_COLORS[color_idx],
                linestyle=style,
                label=COND_NAMES[c] if i == 0 else "",  # Legend only on first plot or specific
                lw=lw,
                alpha=alpha,
            )
            # Mark start
            ax.scatter(
                trajs[c, 0, 0],
                trajs[c, 0, 1],
                trajs[c, 0, 2],
                color=BASE_COLORS[color_idx],
                s=15,
            )

            ax.scatter(
                trajs[c, -1, 0],
                trajs[c, -1, 1],
                trajs[c, -1, 2],
                color=BASE_COLORS[color_idx],
                marker="x",
                s=20,
                alpha=0.6,
            )

        ax.set_title(epoch_name)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if i == 0:
            ax.legend(loc="upper left", fontsize="xx-small", frameon=False)

    plt.tight_layout()
    plt.show()


def plot_pca_single(
    accumulated_data, region, conditions=8, epoch="Choice", cond_names=["Left", "Right"]
):

    print(f"\n--- Visualizing Region: {region} ---")

    epochs_ordered = ["Quiescent", "Stimulus", "Choice"]

    ax = plt.figure(figsize=(5, 5)).add_subplot(projection="3d")
    for i, epoch_name in enumerate(epochs_ordered):
        if epoch_name != epoch:
            continue

        session_matrices = accumulated_data[region][epoch_name]
        if not session_matrices:
            continue

        pop_matrix = np.vstack(session_matrices)
        pca = PCA(n_components=3)
        X_embedded = pca.fit_transform(pop_matrix.T)  # (Total_Time, 3)

        n_bins = int(X_embedded.shape[0] / conditions)  # Divided by conditions conditions
        trajs = X_embedded.reshape(conditions, n_bins, 3)
        for c in range(conditions):

            style = "--"
            alpha = 0.6
            lw = 1.5

            ax.plot(
                trajs[c, :, 0],
                trajs[c, :, 1],
                trajs[c, :, 2],
                linestyle=style,
                label=cond_names[c],
                lw=lw,
                alpha=alpha,
            )
            # Mark start
            ax.scatter(
                trajs[c, 0, 0],
                trajs[c, 0, 1],
                trajs[c, 0, 2],
                s=25,
            )

        ax.set_title(f"{epoch_name}: {region}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper left", fontsize="xx-small", frameon=False)

    plt.tight_layout()
    plt.show()


def plot_pcas_separate_decomposition_adapted(accumulated_data, region, cond_names, colors=None):
    """
    Plots PCA trajectories for a specific region across epochs.
    Adapts automatically to the number of conditions provided in cond_names.

    Args:
        accumulated_data (dict): Data structure containing session matrices.
        region (str): The specific brain region key to access in accumulated_data.
        cond_names (list of str): Names of the conditions (e.g., ['Left', 'Right'] or
                                  ['L_Cong', 'L_Incong', 'R_Cong', 'R_Incong']).
                                  The data is reshaped based on len(cond_names).
        colors (list, optional): List of colors (hex, string, or tuple) matching cond_names.
                                 If None, a default colormap is used.
    """

    n_conditions = len(cond_names)

    if colors is None:

        cmap = plt.get_cmap("tab10") if n_conditions <= 10 else plt.get_cmap("viridis")
        colors = [cmap(i) for i in np.linspace(0, 1, n_conditions)]

    if len(colors) < n_conditions:
        print(
            f"Warning: Provided {len(colors)} colors for {n_conditions} conditions. Cycling colors."
        )
        colors = colors * (n_conditions // len(colors) + 1)

    print(f"\n--- Visualizing Region: {region} ---")

    epochs_ordered = ["Quiescent", "Stimulus", "Choice"]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"{region} | PCA Common Space", fontsize=16)

    for i, epoch_name in enumerate(epochs_ordered):

        if epoch_name not in accumulated_data[region]:
            continue

        session_matrices = accumulated_data[region][epoch_name]
        if not session_matrices:
            continue

        pop_matrix = np.vstack(session_matrices)

        if pop_matrix.shape[1] < 3:
            print(f"Skipping {epoch_name}: Not enough samples for 3 PCA components.")
            continue

        pca = PCA(n_components=3)
        X_embedded = pca.fit_transform(pop_matrix.T)

        total_samples = X_embedded.shape[0]

        if total_samples % n_conditions != 0:
            print(
                f"Error in {epoch_name}: Total samples ({total_samples}) not divisible by # conditions ({n_conditions})."
            )
            continue

        n_bins = int(total_samples / n_conditions)

        trajs = X_embedded.reshape(n_conditions, n_bins, 3)

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        for c in range(n_conditions):

            ax.plot(
                trajs[c, :, 0],
                trajs[c, :, 1],
                trajs[c, :, 2],
                color=colors[c],
                label=cond_names[c] if i == 0 else "",  # Legend only on first subplot
                lw=2,
                alpha=0.8,
            )

            ax.scatter(
                trajs[c, 0, 0],
                trajs[c, 0, 1],
                trajs[c, 0, 2],
                color=colors[c],
                s=20,
            )

            ax.scatter(
                trajs[c, -1, 0],
                trajs[c, -1, 1],
                trajs[c, -1, 2],
                color=colors[c],
                marker="x",
                s=20,
                alpha=0.6,
            )

        ax.set_title(epoch_name)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        if i == 0:
            ax.legend(loc="upper left", fontsize="x-small", frameon=False)

    plt.tight_layout()
    plt.show()


def compute_statistics(list_of_eids):

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    all_stats_list = []
    for eid in tqdm(list_of_eids):
        trials, mask = load_trials_and_mask(one, eid)
        trials = {k: v[mask] for k, v in trials.items()}

        df_stat = compute_animal_stats(trials, eid)
        all_stats_list.append(df_stat)
    return pd.concat(all_stats_list, ignore_index=True)


def run_parallel(task_list, difficulty):

    MAX_WORKERS = 8

    print(f"Found {len(task_list)} sessions. Starting extraction with {MAX_WORKERS} cores...")
    t0 = time.time()

    accumulated_data = {reg: {ep: [] for ep in EPOCHS} for reg in MY_REGIONS}

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
                difficulty=difficulty,
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

    save_path = f"./data/generated/bwm_accumulated_data_correct_{difficulty}_actionkernel.pkl"

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
    # df_all = compute_statistics(list_of_eids)

    # df_all.to_csv("./data/generated/reaction_time_stats.csv", index=False)

    difficulty = 2
    run_parallel(task_list, difficulty)

    difficulty = 3
    run_parallel(task_list, difficulty)

    difficulty = 4
    run_parallel(task_list, difficulty)
