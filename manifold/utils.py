import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import pickle as pkl
from scipy import stats
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from iblatlas.plots import plot_scalar_on_slice
import matplotlib.colors as mcolors


def compute_summary_df(files):
    df = pd.DataFrame(columns=["eid", "region", "score", "median-score", "median-null", "p-value"])
    for fx in files:
        with open(fx, "rb") as f:
            session_data = pkl.load(f)
        eid = fx.rsplit("/")[-1].rsplit("_stim")[0]
        for region in session_data.keys():
            mean_score = session_data[region]["mean_score"]
            median_score = np.nanmedian(session_data[region]["outer_scores"])
            pseudo_score_median = np.nanmedian(session_data[region]["pseudoessions"])

            count = np.sum(session_data[region]["pseudoessions"] >= mean_score)
            n_permutations = len(session_data[region]["pseudoessions"])
            p_value = (count + 1) / (n_permutations + 1)
            df.loc[len(df)] = [eid, region, mean_score, median_score, pseudo_score_median, p_value]
    return df


def compute_significance_by_region(group, alpha_level=0.05, test="fisher"):
    result = pd.Series()
    result["median-score"] = np.nanmedian(group["score"])
    result["std"] = np.nanstd(group["score"])
    result["null_median_of_medians"] = group["median-null"].median()
    if test == "fisher":
        result["pval_combined"] = stats.combine_pvalues(group["p-value"], method="fisher")[1]
    elif test == "wilcoxon":
        stat, pval = stats.wilcoxon(group["score"], group["median-null"], alternative="greater")
        result["pval_combined"] = pval
    else:
        pval = np.nan
    result["frac_sig"] = np.mean(group["p-value"] < alpha_level)
    return result


def compute_complete_df(df):
    df_agg = (
        df.groupby(["region"])
        .apply(lambda x: compute_significance_by_region(x, test="wilcoxon"), include_groups=False)  # type: ignore
        .reset_index()
    )

    _, pvals_combined_corrected, _, _ = multipletests(
        pvals=df_agg["pval_combined"],
        alpha=0.05,
        method="fdr_bh",  # bonferroni?
    )

    df_agg["pval_combined_corrected"] = pvals_combined_corrected
    df_agg["sig_combined_corrected"] = df_agg.pval_combined_corrected < 0.05
    return df_agg


def plot_results(df, df_agg):

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )
    sns.barplot(
        data=df,
        x="region",
        y="score",
        errorbar="sd",
        capsize=0.1,
        alpha=0.5,
        edgecolor="black",
        err_kws={"linewidth": 2},
        ax=ax1,
    )

    sns.stripplot(
        data=df,
        x="region",
        y="score",
        jitter=0.2,
        size=6,
        alpha=0.8,
        linewidth=1,
        dodge=False,
        ax=ax1,
    )

    ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title("Decoding Accuracy", fontsize=16, pad=15)
    ax1.set_ylabel("Balanced Accuracy Score", fontsize=12)
    ax1.set_xlabel("")  # Remove x-label here since it's shared with the bottom plot
    ax1.set_ylim(0.3, 1.0)

    sns.barplot(
        data=df_agg,
        x="region",
        y="frac_sig",
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
        ax=ax2,
    )

    ax2.set_ylabel("Fraction Significant", fontsize=12)
    ax2.set_xlabel("Regions", fontsize=12)
    ax2.set_ylim(0, 1.0)  # Fraction goes from 0 to 1

    # Rotate the x-axis labels on the bottom plot so they don't overlap
    ax2.tick_params(axis="x", rotation=90)

    sns.despine()
    plt.tight_layout()


def plot_topdown(regions, scores, epoch, cmap="jet"):

    # 1. Your actual regions and R2 scores

    fig, ax = plt.subplots(figsize=(6, 6))
    fig, ax, cbar = plot_scalar_on_slice(  # type: ignore
        regions=regions,
        values=scores,
        slice="top",
        vector=True,
        mapping="Beryl",
        hemisphere="left",
        background="boundary",
        empty_color="white",
        ax=ax,
        cmap=cmap,
        show_cbar=True,
        clevels=[0.0, 1],
    )

    cbar.ax.set_title("balanced accuracy", fontsize=12, pad=12)
    cbar.outline.set_visible(False)  # type: ignore

    ax.set_title(f"WFI-{epoch}", fontsize=14, pad=15)
    ax.axis("off")


def get_trial_masks(trials):
    """
    Returns boolean masks for 2 conditions (Incongruent, correct and incorrect).
    """
    masks = {}

    is_L_block = trials["probabilityLeft"] == 0.8
    is_R_block = trials["probabilityLeft"] == 0.2

    has_contrast_L = ~np.isnan(trials["contrastLeft"])
    has_contrast_R = ~np.isnan(trials["contrastRight"])
    is_correct = trials["feedbackType"] == 1
    is_error = trials["feedbackType"] == -1

    # incongruent correct
    right_stimulus_incong_correct = has_contrast_R & is_L_block & is_correct
    left_stimulus_incong_correct = has_contrast_L & is_R_block & is_correct
    right_stimulus_incong_incorrect = has_contrast_R & is_L_block & is_error
    left_stimulus_incong_incorrect = has_contrast_L & is_R_block & is_error

    incong_correct = right_stimulus_incong_correct | left_stimulus_incong_correct
    incong_incorrect = right_stimulus_incong_incorrect | left_stimulus_incong_incorrect

    masks["Incongruent_correct"] = incong_correct
    masks["Incongruent_incorrect"] = incong_incorrect

    return masks, list(masks.keys())
