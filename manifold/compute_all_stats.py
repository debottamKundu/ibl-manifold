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


def compute_stats_over_pseudo_ids(group, n_pseudo=2):
    """Aggregate info over pseudo_ids."""
    result = pd.Series(dtype="float64")

    eid = group.name[1]
    region = group.name[2]

    a = group.loc[group["pseudo_id"] == -1, "score_test"].values
    b = group.loc[group["pseudo_id"] > 0, "score_test"].values
    if len(b) != n_pseudo:
        print(f"result for {eid}-{region} does not contain {n_pseudo} pseudo-sessions")
    result["n_pseudo"] = len(b)
    if (len(a) == 0) or (len(b) != n_pseudo):
        result["score"] = np.nan
        result["p-value"] = np.nan
        result["median-null"] = np.nan
        result["n_trials"] = np.nan
    else:
        result["score"] = a[0]
        result["p-value"] = np.mean(np.concatenate([np.array(b), [a[0]]]) >= a[0])
        result["median-null"] = np.median(b)
        # collect number of trials, only from real session
        c = group.loc[group["pseudo_id"] == -1, "n_trials"].values
        n_trials = np.unique(c)
        assert len(n_trials) == 1, "n_trials vals do not agree across all runs"
        result["n_trials"] = int(n_trials[0])
    return result


def reformat_df(df):
    """Compute mean over runs and median of null distribution; compute p-value."""

    df_tmp = df.groupby(
        ["subject", "eid", "region", "N_units", "pseudo_id", "n_trials"], as_index=False
    )["score_test"].mean()

    df_new = (
        df_tmp.groupby(["subject", "eid", "region", "N_units"])
        .apply(lambda x: compute_stats_over_pseudo_ids(x), include_groups=False)
        .reset_index()
    )
    df_new = df_new.rename(columns={"N_units": "n_units"})

    return df_new


def significance_by_region(group):
    result = pd.Series()
    # only get p-values for sessions with min number of trials
    if "n_trials" in group:
        trials_mask = group["n_trials"] >= MIN_TRIALS
    else:
        trials_mask = np.ones(group.shape[0]).astype("bool")
    pvals = group.loc[trials_mask, "p-value"].values
    pvals = np.array([p if p > 0 else 1.0 / (N_PSEUDO + 1) for p in pvals])
    # count number of good sessions
    n_sessions = len(pvals)
    result["n_sessions"] = n_sessions
    # only compute combined p-value if there are enough sessions
    if n_sessions < MIN_SESSIONS_PER_REGION:
        result["pval_combined"] = np.nan
        result["n_units_mean"] = np.nan
        result["values_std"] = np.nan
        result["values_median"] = np.nan
        result["null_median_of_medians"] = np.nan
        result["valuesminusnull_median"] = np.nan
        result["frac_sig"] = np.nan
        result["values_median_sig"] = np.nan
        result["sig_combined"] = np.nan
    else:
        scores = group.loc[trials_mask, "score"].values
        result["pval_combined"] = stats.combine_pvalues(pvals, method="fisher")[1]
        result["n_units_mean"] = group.loc[trials_mask, "n_units"].mean()
        result["values_std"] = np.std(scores)
        result["values_median"] = np.median(scores)
        result["null_median_of_medians"] = group.loc[trials_mask, "median-null"].median()
        result["valuesminusnull_median"] = (
            result["values_median"] - result["null_median_of_medians"]
        )
        result["frac_sig"] = np.mean(pvals < ALPHA_LEVEL)
        result["values_median_sig"] = np.median(scores[pvals < ALPHA_LEVEL])
        result["sig_combined"] = result["pval_combined"] < ALPHA_LEVEL
    return result


if __name__ == "__main__":

    MIN_UNITS = 5
    MIN_TRIALS = 50
    MIN_SESSIONS_PER_REGION = 2
    ALPHA_LEVEL = 0.05
    Q_LEVEL = 0.05
    N_PSEUDO = 100

    data_dir = "./data/ephys/"
    finished = glob(data_dir + "/*/*/*.pqt")

    resultsdf = load_and_combine_df(finished)

    resultsdf.to_parquet("./data/ephys/collected_results_stage1.pqt")

    df_collected = reformat_df(df=resultsdf)

    df_collected.to_parquet("./data/ephys/collected_results_stage2.pqt")

    df2 = (
        df_collected[df_collected.n_units >= MIN_UNITS]
        .groupby(["region"])
        .apply(lambda x: significance_by_region(x), include_groups=False)
        .reset_index()
    )

    mask = ~df2["pval_combined"].isna()
    _, pvals_combined_corrected, _, _ = multipletests(
        pvals=df2.loc[mask, "pval_combined"],
        alpha=Q_LEVEL,
        method="fdr_bh",
    )
    df2.loc[mask, "pval_combined_corrected"] = pvals_combined_corrected
    df2.loc[:, "sig_combined_corrected"] = df2.pval_combined_corrected < Q_LEVEL

    df2.to_parquet("./data/ephys/collected_results_stage3.pqt")
