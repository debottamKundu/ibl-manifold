import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

from one.api import ONE
from brainbox.io.one import SessionLoader
from iblatlas.atlas import BrainRegions
from sklearn.metrics import balanced_accuracy_score
import traceback

from brainwidemap.bwm_loading import (
    bwm_query,
    load_good_units,
    load_trials_and_mask,
    merge_probes,
    bwm_units,
)
from brainwidemap.decoding.functions.decoding import fit_eid
from brainwidemap.decoding.functions.process_targets import optimal_Bayesian
from manifold.feedback_settings_template import params
from decoding.functions.utils import check_config_decoding

config = check_config_decoding()
MY_REGIONS = config["prior_regions"]

try:
    from behavior_models.models import ActionKernel

    HAS_BEH_MODELS = True
except ImportError:
    HAS_BEH_MODELS = False

"""
Advanced Custom Decoding Script for Cluster Execution
-----------------------------------------------------
Features:
- Parallel execution support via EID index.
- Balanced weighting for Logistic Regression.
- Pseudosession generation using pre-computed ActionKernel models.
- Performance analysis on Congruent vs. Incongruent trials.

Congruent: Stimulus side == Block side.
"""

# Update model dispatcher if models are available
model_dispatcher = {optimal_Bayesian: "optBay", None: "oracle"}
if HAS_BEH_MODELS:
    model_dispatcher[ActionKernel] = ActionKernel.name





def run_decoding_for_eid(
    eid, regions, time_window, align_time, n_pseudo, results_dir, behfit_path
):
    one = ONE(base_url="https://openalyx.internationalbrainlab.org")
    br = BrainRegions()

    # Override params
    params["target"] = "feedback"
    params["align_time"] = align_time
    params["time_window"] = time_window
    params["binsize"] = time_window[1] - time_window[0]
    params["n_pseudo"] = n_pseudo
    params["balanced_weight"] = True  # Handle unbalanced classes

    # Use ActionKernel for pseudosessions if available
    if HAS_BEH_MODELS:
        params["model"] = ActionKernel
        params["modeldispatcher"] = model_dispatcher
        params["behfit_path"] = Path(behfit_path)
    else:
        print(
            "Warning: behavior_models not found. Pseudosessions will use default trial sequence."
        )
        params["model"] = None

    # Setup paths
    params["neuralfit_path"] = results_dir.joinpath("decoding", "results", "neural")
    params["neuralfit_path"].mkdir(parents=True, exist_ok=True)
    params["behfit_path_save"] = results_dir.joinpath("decoding", "results", "behavioral")
    params["behfit_path_save"].mkdir(parents=True, exist_ok=True)
    params["add_to_saving_path"] = f"_binsize={1000 * params['binsize']}_custom_feedback"

    print(f"Processing session {eid}...")
    bwm_df = bwm_query(one)
    sess_df = bwm_df[bwm_df["eid"] == eid]
    if sess_df.empty:
        print(f"EID {eid} not found.")
        return

    pids = sess_df["pid"].tolist()
    probe_names = sess_df["probe_name"].tolist()
    subject = sess_df["subject"].iloc[0]

    # Load trials
    sess_loader = SessionLoader(one=one, eid=eid)
    sess_loader.load_trials()
    trials_df, trials_mask = load_trials_and_mask(
        one=one,
        eid=eid,
        sess_loader=sess_loader,
        min_rt=params["min_rt"],  # maybe change here
        max_rt=params["max_rt"],  # also here
        exclude_nochoice=True,
        exclude_unbiased=True,  # we only need feedback for incongruent ideally
    )
    params["trials_mask_diagnostics"] = [trials_mask]

    # Load units and filter by regions
    clusters_list = []
    spikes_list = []
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters = load_good_units(one, pid, eid=eid, pname=probe_name)
        tmp_clusters["pid"] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)

    if params["merged_probes"] and len(clusters_list) > 1:
        spikes, clusters = merge_probes(spikes_list, clusters_list)
    else:
        spikes, clusters = spikes_list[0], clusters_list[0]

    beryl_acronyms = br.acronym2acronym(clusters["acronym"], mapping="Beryl")
    cluster_mask = np.isin(beryl_acronyms, regions)

    if not np.any(cluster_mask):
        print(f"No units in {regions}.")
        return

    valid_cluster_indices = clusters.index[cluster_mask].values
    filtered_clusters = clusters[cluster_mask].copy()
    spike_mask = np.isin(spikes["clusters"], valid_cluster_indices)
    filtered_spikes = {k: v[spike_mask] for k, v in spikes.items()}
    old_to_new = {old: new for new, old in enumerate(valid_cluster_indices)}
    filtered_spikes["clusters"] = np.array([old_to_new[c] for c in filtered_spikes["clusters"]])
    filtered_clusters.reset_index(drop=True, inplace=True)

    neural_dict = {
        "spk_times": filtered_spikes["times"],
        "spk_clu": filtered_spikes["clusters"],
        "clu_regions": filtered_clusters["acronym"],
        "clu_qc": {k: np.asarray(v) for k, v in filtered_clusters.to_dict("list").items()},
        "clu_df": filtered_clusters,
    }
    metadata = {"subject": subject, "eid": eid, "probe_name": "merged"}

    # Run decoding (including pseudosessions)
    pseudo_ids = np.concatenate([-np.ones(1), np.arange(n_pseudo) + 1]).astype(int)

    # print(params)

    print(f"Running fits for {len(pseudo_ids)} sessions...")
    filenames, _ = fit_eid(
        neural_dict=neural_dict,
        trials_df=trials_df,
        trials_mask=trials_mask,
        metadata=metadata,
        pseudo_ids=pseudo_ids,
        dlc_dict=None,
        **params,
    )

    # Post-processing: Compute performance on congruent/incongruent trials
    print("Computing subset performances...")
    for filename in filenames:
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # fit_result is a list of results (one per run)
        for fit_res in data["fit"]:
            subset_perf = compute_subset_performance(fit_res, trials_df, trials_mask)
            fit_res["subset_perf"] = subset_perf

        # Save back the updated results
        with open(filename, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":

    n_pseudo = 2
    results_dir = "./cluster_results"
    behfit_path = "./results_behavioral"

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    print("Querying BWM Units...")

    units_df = bwm_units(one)
    flattened_regions = [item for sublist in MY_REGIONS for item in sublist]
    relevant_pids = units_df[units_df["Beryl"].isin(flattened_regions)]["pid"].unique()

    bwm_df = bwm_query(one)

    subset_df = bwm_df[bwm_df["pid"].isin(relevant_pids)]
    list_of_eids = subset_df["eid"].unique()

    TIME_WINDOW = (-0.4, -0.1)
    ALIGN_TIME = "stimOn_times"

    print(len(list_of_eids))
    for index in range(len(list_of_eids)):
        try:
            run_decoding_for_eid(
                list_of_eids[index],
                MY_REGIONS,
                TIME_WINDOW,
                ALIGN_TIME,
                n_pseudo,
                Path(results_dir),
                behfit_path,
            )
        except Exception as e:
            traceback.print_exc()
        break

#  To resume this session: gemini --resume 09909a69-29c3-4808-aff5-329849f93dd1
