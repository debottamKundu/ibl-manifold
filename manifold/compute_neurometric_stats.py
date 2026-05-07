import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from one.api import ONE
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
from glob import glob

from tqdm import tqdm


def load_and_combine_df(finished):
    expected_columns = [
        "subject",
        "eid",
        "probe",
        "region",
        "N_units",
        "pseudo_id",
        "run_id",
        "score_test",
        "n_trials",
        "low_slope",
        "high_slope",
    ]

    results_dfs = []
    failed_load = 0

    for fn in tqdm(finished):
        if os.path.isdir(fn):
            continue
        try:
            df_file = pd.read_parquet(fn)
            df_file = df_file.rename(columns={"R2_test": "score_test", "nb_trials": "n_trials"})
            cols_to_keep = [c for c in expected_columns if c in df_file.columns]
            df_file = df_file[cols_to_keep]

            results_dfs.append(df_file)

        except Exception as e:
            print(f"Failed loading: {fn}")
            print(e)
            failed_load += 1

    print("loading of %i files failed" % failed_load)

    if len(results_dfs) > 0:
        resultsdf = pd.concat(results_dfs, ignore_index=True)

    return resultsdf


def reformat_df(df):
    """Compute mean over runs and median of null distribution; compute p-value."""

    df_tmp = df.groupby(
        ["subject", "eid", "region", "N_units", "pseudo_id", "n_trials"], as_index=False
    )[["score_test", "low_slope", "high_slope"]].mean()

    df_new = (
        df_tmp.groupby(["subject", "eid", "region", "N_units"])
        .apply(lambda x: compute_stats_over_pseudo_ids(x), include_groups=False)
        .reset_index()
    )
    df_new = df_new.rename(columns={"N_units": "n_units"})

    return df_new


def compute_stats_over_pseudo_ids(group, n_pseudo=200):
    """Aggregate info over pseudo_ids."""
    result = pd.Series(dtype="float64")

    eid = group.name[1]
    region = group.name[2]

    real_data = group[group["pseudo_id"] == -1]
    null_data = group[group["pseudo_id"] > 0]

    a_score = real_data["score_test"].values
    b_score = null_data["score_test"].values

    if len(b_score) != n_pseudo:
        print(f"result for {eid}-{region} does not contain {n_pseudo} pseudo-sessions")

    result["n_pseudo"] = len(b_score)

    if (len(a_score) == 0) or (len(b_score) != n_pseudo):
        result["score"] = np.nan
        result["p-value"] = np.nan
        result["median-null"] = np.nan
        result["n_trials"] = np.nan
        result["low_slope"] = np.nan
        result["high_slope"] = np.nan
        result["slope_diff_pval"] = np.nan  # <-- New metric
    else:

        result["score"] = a_score[0]
        result["p-value"] = np.mean(
            np.concatenate([np.array(b_score), [a_score[0]]]) >= a_score[0]
        )
        result["median-null"] = np.median(b_score)

        c = real_data["n_trials"].values
        n_trials = np.unique(c)
        assert len(n_trials) == 1, "n_trials vals do not agree across all runs"
        result["n_trials"] = int(n_trials[0])

        a_low = real_data["low_slope"].values[0]
        a_high = real_data["high_slope"].values[0]
        result["low_slope"] = a_low
        result["high_slope"] = a_high

        real_diff = a_high - a_low
        null_diffs = null_data["high_slope"].values - null_data["low_slope"].values
        all_diffs = np.concatenate([null_diffs, [real_diff]])
        result["slope_diff_pval"] = np.mean(np.abs(all_diffs) >= np.abs(real_diff))
        result["slope_diff_pval_oneside"] = np.mean(all_diffs >= real_diff)

    return result


if __name__ == "__main__":

    MIN_UNITS = 5
    MIN_TRIALS = 50
    MIN_SESSIONS_PER_REGION = 2
    ALPHA_LEVEL = 0.05
    Q_LEVEL = 0.05
    N_PSEUDO = 100

    data_dir = "./data/ephys_neurometric/"
    finished = glob(data_dir + "/*/*/*.pqt")

    print(f"Found {len(finished)} files. Loading data...")
    resultsdf = load_and_combine_df(finished)

    resultsdf.to_parquet("./data/ephys/collected_results_stage1.pqt")

    print("Reformatting dataframe and computing session-level statistics...")
    df_collected = reformat_df(df=resultsdf)

    df_collected.to_parquet("./data/ephys/collected_results_stage2.pqt")

    print("Computing region-level summaries for R2 task significance...")
