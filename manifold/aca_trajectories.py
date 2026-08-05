

import os
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
from one.api import ONE
from brainwidemap import bwm_units, load_trials_and_mask, load_good_units
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions



def load_ancillary_data(base_dir):
    """
    Load behavioral states, motivation scalar, and action kernel data.
    Looks in standard locations under base_dir.
    """
    
    engagement_df = pd.read_parquet('./data/external/merged_behavioral_and_states.pqt') # this is the glm engagement

    with open('./data/external/all_eids_engagement.pkl','rb') as f:
            scalar_motivation = pkl.load(f)    
    with open('./data/external/action_kernel.pkl','rb') as f:
        akernel = pkl.load(f)
    return engagement_df, scalar_motivation, akernel


def prepare_session_trials(
    one,
    eid,
    engagement_df,
    scalar_motivation,
    akernel,
):
    """
    Load trials and apply filters, merging action kernel and motivation signals
    """
    trials_df, mask = load_trials_and_mask(one, eid)

    if engagement_df is not None:
        animal_engagement = engagement_df[engagement_df["eid"] == eid].reset_index(drop=True)
        if len(animal_engagement) == len(trials_df):
            cols_to_merge = [c for c in ["p_state1", "p_state2", "signed_contrast", "rewarded"] if c in animal_engagement.columns]
            trials_df = trials_df.merge(animal_engagement[cols_to_merge], left_index=True, right_index=True)

    if akernel is not None and eid in akernel:
        akernel_df = akernel[eid]
        if len(akernel_df) == len(trials_df):
            if "prior" in akernel_df.columns:
                trials_df["akernel_prior"] = akernel_df["prior"].values
            if "prediction_error_right" in akernel_df.columns:
                trials_df["pe_right"] = akernel_df["prediction_error_right"].values
            if "prediction_error_left" in akernel_df.columns:
                trials_df["pe_left"] = akernel_df["prediction_error_left"].values

    if scalar_motivation is not None and eid in scalar_motivation:
        stimulus_engage = scalar_motivation[eid]
        if len(stimulus_engage) == len(trials_df):
            trials_df["motivation_scalar"] = stimulus_engage

    if "signed_contrast" not in trials_df.columns:
        trials_df["signed_contrast"] = np.nan_to_num(trials_df["contrastLeft"]) - np.nan_to_num(trials_df["contrastRight"])
    trials_df["eid"] = eid
    trials_df["trial_id"] = trials_df.index

    masked_trials_df = trials_df[mask].copy().reset_index(drop=True)
    return trials_df, masked_trials_df


def process_session_spikes(
    one,
    eid,
    session_units,
    masked_trials_df,
    pre_stim_window=(0.25, 0.0), 
    traj_window=(0.25, 0.1),
    bin_size= 0.01, # 10ms bin size
):
    """
    Extract pre-stimulus spike counts and stimulus-locked trajectories for ACAd/ACAv units.
    """
    align_times = masked_trials_df["stimOn_times"].values
    if len(align_times) == 0:
        return None

    
    unique_pids = session_units["pid"].unique()
    all_units_meta = []
    all_pre_counts = []  # list of (n_trials, n_probe_units)
    all_trajectories = [] # list of (n_trials, n_probe_units, n_bins)
    time_bins_saved = None

    for pid in unique_pids:
        probe_aca_units = session_units[session_units["pid"] == pid].copy()
        probe_aca_units = probe_aca_units.sort_values(by="cluster_id").reset_index(drop=True)
        target_cluster_ids = np.sort(probe_aca_units["cluster_id"].values)

        try:
            spikes, clusters = load_good_units(one, pid=pid, eid=eid)
        except Exception as e:
            print(f"Warning: Failed to load spikes for pid {pid} in eid {eid}: {e}")
            continue

        all_spike_ids = clusters["cluster_id"][spikes["clusters"]]
        spike_mask = np.isin(all_spike_ids, target_cluster_ids)
        region_spike_times = spikes["times"][spike_mask]
        region_spike_ids = all_spike_ids[spike_mask]

        pre_count_binned, _ = bin_spikes2D(
            region_spike_times,
            region_spike_ids,
            target_cluster_ids,
            align_times,
            pre_time=pre_stim_window[0],
            post_time=pre_stim_window[1],
            bin_size=pre_stim_window[0] - pre_stim_window[1],
        )
        pre_counts = np.squeeze(pre_count_binned, axis=-1)

        traj_binned, bintimes = bin_spikes2D(
            region_spike_times,
            region_spike_ids,
            target_cluster_ids,
            align_times,
            pre_time=traj_window[0],
            post_time=traj_window[1],
            bin_size=bin_size,
        )
        
        traj_rates = traj_binned / bin_size

        if time_bins_saved is None:
            time_bins_saved = bintimes

        all_units_meta.append(probe_aca_units)
        all_pre_counts.append(pre_counts)
        all_trajectories.append(traj_rates)

    if not all_units_meta:
        return None

    combined_units_meta = pd.concat(all_units_meta, ignore_index=True)
    combined_pre_counts = np.concatenate(all_pre_counts, axis=1)      
    combined_trajectories = np.concatenate(all_trajectories, axis=1)  

    return {
        "eid": eid,
        "trials": masked_trials_df,
        "units": combined_units_meta,
        "pre_stim_counts": combined_pre_counts,     # (n_trials, n_units)
        "trajectories": combined_trajectories,       # (n_trials, n_units, n_bins)
        "time_bins": time_bins_saved,                # 1D array of bin centers relative to stimOn
    }


def format_to_dataframe(session_data_list):
    """
    Format extracted session data into a unified trial-level pandas DataFrame.
    """
    trial_records = []

    for s_data in session_data_list:
        eid = s_data["eid"]
        trials = s_data["trials"]
        units = s_data["units"]
        pre_counts = s_data["pre_stim_counts"]
        trajs = s_data["trajectories"]
        unit_ids = [f"{row.pid}_{row.cluster_id}" for _, row in units.iterrows()]
        beryl_regions = units["Beryl"].tolist()

        for t_idx, (_, trial) in enumerate(trials.iterrows()):
            rec = trial.to_dict()
            rec["eid"] = eid
            rec["n_aca_units"] = len(unit_ids)
            rec["unit_ids"] = unit_ids
            rec["unit_regions"] = beryl_regions
            rec["pre_stim_spike_counts"] = pre_counts[t_idx, :]
            rec["trajectories"] = trajs[t_idx, :, :]
            trial_records.append(rec)

    return pd.DataFrame(trial_records)


def run_pipeline(
    base_dir,
    single_eid,
    output_dir,
    save_parquet,
    save_pickle,
):
    """
    Run the full extraction pipeline for ACAd / ACAv units.
    """
    if output_dir is None:
        output_dir = base_dir / "data" / "generated" / "aca_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    print("Loading bwm_units...")
    units_df = bwm_units(one)
    aca_units_all = units_df[units_df["Beryl"].isin(["ACAd", "ACAv"])].copy()

    unique_eids = aca_units_all["eid"].unique()
    print(f"Total sessions with ACAd or ACAv units: {len(unique_eids)}")

    if single_eid:
        if single_eid not in unique_eids:
            print(f"Warning: Specified eid {single_eid} not in ACAd/ACAv list. Checking if valid...")
        eids_to_process = [single_eid]
    else:
        eids_to_process = list(unique_eids)

    print(f"Processing {len(eids_to_process)} session(s)...")

 
    engagement_df, scalar_motivation, akernel = load_ancillary_data(base_dir)

    session_results = []
    for i, eid in enumerate(eids_to_process):
        print(f"[{i+1}/{len(eids_to_process)}] Processing eid: {eid}")
        session_units = aca_units_all[aca_units_all["eid"] == eid]
        if len(session_units) == 0:
            print(f"  No ACAd/ACAv units found for eid {eid}, skipping.")
            continue

        try:
            _, masked_trials_df = prepare_session_trials(
                one, eid, engagement_df, scalar_motivation, akernel
            )
            print(f"  Valid masked trials: {len(masked_trials_df)}, ACAd/ACAv units: {len(session_units)}")

            res = process_session_spikes(
                one, eid, session_units, masked_trials_df,
                pre_stim_window=(0.25, 0.0),
                traj_window=(0.25, 0.1),
                bin_size=0.01,
            )
            if res is not None:
                session_results.append(res)
                print(f"  Successfully extracted spikes & trajectories for {eid}.")
        except Exception as e:
            print(f"  Error processing eid {eid}: {e}")

    out_dict = {
        "sessions": session_results,
        "time_bins": session_results[0]["time_bins"] if session_results else None,
        "pre_stim_window": (-0.25, 0.0),
        "traj_window": (-0.25, 0.1),
        "bin_size": 0.01,
    }

    if save_pickle:
        pkl_path = output_dir / ("single_session_aca_trajectories.pkl" if single_eid else "all_aca_trajectories.pkl")
        with open(pkl_path, "wb") as f:
            pkl.dump(out_dict, f)
        print(f"Saved results pickle to: {pkl_path}")

    df_combined = format_to_dataframe(session_results)
    if save_parquet and len(df_combined) > 0:
        df_pkl_path = output_dir / ("single_session_aca_df.pkl" if single_eid else "all_aca_df.pkl")
        df_combined.to_pickle(df_pkl_path)
        print(f"Saved trial dataframe pickle to: {df_pkl_path}")

    return out_dict


def main():
    
    
    
    base_dir = "./"
    out_dir = "./data/generated/aca_trajectories/"
    single_eid = False
    run_pipeline(base_dir=base_dir, single_eid=single_eid, output_dir=out_dir, save_parquet=True, save_pickle=True)


if __name__ == "__main__":
    main()
