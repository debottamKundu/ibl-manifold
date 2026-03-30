import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns
import pickle as pkl
import statsmodels.formula.api as smf
import re


def load_trials_and_plot(
    trial_files, action_kernel_files, eid, window_size=10, state_idx="engagement_state", plot=False
):

    df = trial_files[eid]
    df_ak = action_kernel_files[eid]
    df["akernel"] = df_ak["prior"]

    df_alt = df.copy()
    df_alt = df_alt.reset_index(drop=True)
    df_alt["rt_sec"] = df_alt["response_times"] - df_alt["stimOn_times"]

    df_alt = df_alt[(df_alt["rt_sec"] >= 0.08) & (df_alt["rt_sec"] <= 2.0)].copy()
    # df_alt = df_alt[(df_alt["rt_sec"] >= 0.08)].copy()
    # df_alt["contrastRight"] = df_alt["contrastRight"].fillna(0)
    # df_alt["contrastLeft"] = df_alt["contrastLeft"].fillna(0)
    df_alt["stimulus_side"] = np.where(df_alt["contrastRight"].notna(), 1.0, -1.0)
    df_alt["absolute_contrast"] = df_alt[["contrastRight", "contrastLeft"]].max(axis=1)

    df_alt["engagement_state"] = df_alt["glm-hmm_2"].apply(lambda x: np.argmax(x))
    # df_alt["State_Label"] = df_alt["engagement_state"].map({1: "engaged", 0: "disengaged"})

    df_alt["expected_side"] = np.where(df_alt["akernel"] > 0.5, -1.0, 1.0)
    df_alt["internal_congruent"] = df_alt["expected_side"] == df_alt["stimulus_side"]
    df_alt["Congruence_Label"] = df_alt["internal_congruent"].map(
        {True: "Congruent", False: "Incongruent"}
    )
    df_alt["signed_contrast"] = df_alt["absolute_contrast"] * df_alt["stimulus_side"]

    df_alt["Prior_Label"] = df_alt["expected_side"].map(
        {-1.0: "Expects Left", 1.0: "Expects Right"}
    )
    ordered_contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
    contrast_mapping = {val: i for i, val in enumerate(ordered_contrasts)}
    df_alt["contrast_ordinal"] = df_alt["signed_contrast"].map(contrast_mapping)

    df_alt["is_correct"] = (df_alt["feedbackType"] == 1.0).astype(int)
    engaged_threshold = 0.60

    df_alt["hmm_state_label"] = df_alt["engagement_state"].map({1.0: "engaged", 0.0: "disengaged"})
    df_alt["hmm_state_label"] = df_alt["hmm_state_label"].fillna("engaged")

    df_alt["rolling_accuracy"] = (
        df_alt["is_correct"]
        .shift(1)
        .ewm(span=window_size, min_periods=1)
        .mean()
        # rolling(window=window_size, min_periods=1).mean()
    )

    df_alt["proxy_state_label"] = np.where(
        df_alt["rolling_accuracy"] >= engaged_threshold, "engaged", "disengaged"
    )
    df_alt["proxy_state_label"] = df_alt["proxy_state_label"].fillna("engaged")

    df_correct = df_alt[df_alt["feedbackType"] == 1.0].copy()
    df_incorrect = df_alt[df_alt["feedbackType"] == -1.0].copy()

    df_alt["is_correct"] = (df_alt["feedbackType"] == 1.0).astype(int)

    # state_idx = "proxy_state_label"
    if plot:
        plt.figure(figsize=(6, 5))

        sns.barplot(
            data=df_alt,
            x=state_idx,
            y="is_correct",
            # order=[0, 1],  # Forces engaged to be on the left
            # palette={"engaged": "#4C72B0", "disengaged": "#C44E52"},
            capsize=0.1,
            errorbar=("ci", 95),
        )

        plt.title("Proportion Correct by Engagement State", pad=15)
        plt.ylabel("Fraction Correct")
        plt.xlabel("")  # Leave blank for cleaner look
        plt.ylim(0, 1.05)  # Lock y-axis from 0 to 1
        plt.axhline(0.5, color="black", linestyle="--", alpha=0.5)  # Add chance-level line

        sns.despine()
        plt.tight_layout()
        plt.show()

    return df_correct, df_incorrect, df_alt


def plot_overall_rt(df_correct):
    ordered_contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
    palette = {"Expects Left": "#D9534F", "Expects Right": "#5CB85C"}

    df_counts = (
        df_correct.groupby(["contrast_ordinal", "Prior_Label"]).size().reset_index(name="n_trials")
    )

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    sns.lineplot(
        ax=axes[0],
        data=df_correct,
        x="contrast_ordinal",
        y="rt_sec",
        hue="Prior_Label",
        palette=palette,
        estimator=np.median,
        errorbar=("ci", 95),
        marker="o",
        linewidth=2.5,
    )
    axes[0].set_title("Overall Reaction Time (Correct Trials)")
    axes[0].set_ylabel("Reaction Time (s)")
    axes[0].axvline(4, color="black", linestyle="--", alpha=0.3)

    sns.lineplot(
        ax=axes[1],
        data=df_counts,
        x="contrast_ordinal",
        y="n_trials",
        hue="Prior_Label",
        palette=palette,
        marker="o",
        linestyle="--",
        linewidth=1.5,
        legend=False,
    )
    axes[1].set_ylabel("Trial Count")
    axes[1].axvline(4, color="black", linestyle="--", alpha=0.3)
    axes[1].set_xticks(range(len(ordered_contrasts)))
    axes[1].set_xticklabels(ordered_contrasts, rotation=45)
    axes[1].set_xlabel("Signed Contrast")
    axes[1].set_xlim(-0.5, len(ordered_contrasts) - 0.5)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_rt_by_engagement_correct(df_correct, state_idx="State_Label"):
    ordered_contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
    palette = {"Expects Left": "#D9534F", "Expects Right": "#5CB85C"}
    states = ["engaged", "disengaged"]

    df_counts = (
        df_correct.groupby(["contrast_ordinal", "Prior_Label", state_idx])
        .size()
        .reset_index(name="n_trials")
    )

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("Reaction Time by Engagement (Correct Trials)", y=1.02, fontsize=14)

    for col_idx, state in enumerate(states):
        rt_data = df_correct[df_correct[state_idx] == state]
        count_data = df_counts[df_counts[state_idx] == state]

        sns.lineplot(
            ax=axes[0, col_idx],
            data=rt_data,
            x="contrast_ordinal",
            y="rt_sec",
            hue="Prior_Label",
            palette=palette,
            estimator=np.median,
            errorbar=("ci", 95),
            marker="o",
            linewidth=2.5,
            legend=(col_idx == 0),
        )
        axes[0, col_idx].set_title(f"{state} State")
        axes[0, col_idx].set_ylabel("Reaction Time (s)" if col_idx == 0 else "")
        axes[0, col_idx].axvline(4, color="black", linestyle="--", alpha=0.3)

        sns.lineplot(
            ax=axes[1, col_idx],
            data=count_data,
            x="contrast_ordinal",
            y="n_trials",
            hue="Prior_Label",
            palette=palette,
            marker="o",
            linestyle="--",
            linewidth=1.5,
            legend=False,
        )
        axes[1, col_idx].set_ylabel("Trial Count" if col_idx == 0 else "")
        axes[1, col_idx].axvline(4, color="black", linestyle="--", alpha=0.3)

        axes[1, col_idx].set_xticks(range(len(ordered_contrasts)))
        axes[1, col_idx].set_xticklabels(ordered_contrasts, rotation=45)
        axes[1, col_idx].set_xlabel("Signed Contrast")
        axes[1, col_idx].set_xlim(-0.5, len(ordered_contrasts) - 0.5)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_rt_by_engagement_incorrect(df_incorrect, state_idx="State_Label"):
    ordered_contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
    palette = {"Expects Left": "#D9534F", "Expects Right": "#5CB85C"}
    states = ["engaged", "disengaged"]

    df_counts = (
        df_incorrect.groupby(["contrast_ordinal", "Prior_Label", state_idx])
        .size()
        .reset_index(name="n_trials")
    )

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        "Reaction Time by Engagement (INCORRECT Trials)", y=1.02, fontsize=14, color="darkred"
    )

    for col_idx, state in enumerate(states):
        rt_data = df_incorrect[df_incorrect[state_idx] == state]
        count_data = df_counts[df_counts[state_idx] == state]
        sns.lineplot(
            ax=axes[0, col_idx],
            data=rt_data,
            x="contrast_ordinal",
            y="rt_sec",
            hue="Prior_Label",
            palette=palette,
            estimator=np.median,
            errorbar=None,  # Removed error bars for low-N incorrect trials
            marker="o",
            linewidth=2.5,
            legend=(col_idx == 0),
        )
        axes[0, col_idx].set_title(f"{state} State")
        axes[0, col_idx].set_ylabel("Reaction Time (s)" if col_idx == 0 else "")
        axes[0, col_idx].axvline(4, color="black", linestyle="--", alpha=0.3)
        sns.lineplot(
            ax=axes[1, col_idx],
            data=count_data,
            x="contrast_ordinal",
            y="n_trials",
            hue="Prior_Label",
            palette=palette,
            marker="o",
            linestyle="--",
            linewidth=1.5,
            legend=False,
        )
        axes[1, col_idx].set_ylabel("Trial Count" if col_idx == 0 else "")
        axes[1, col_idx].axvline(4, color="black", linestyle="--", alpha=0.3)
        axes[1, col_idx].set_xticks(range(len(ordered_contrasts)))
        axes[1, col_idx].set_xticklabels(ordered_contrasts, rotation=45)
        axes[1, col_idx].set_xlabel("Signed Contrast")
        axes[1, col_idx].set_xlim(-0.5, len(ordered_contrasts) - 0.5)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_traces(df_all, threshold):
    plt.figure(figsize=(15, 5))

    df_all["trial_num"] = np.arange(len(df_all))
    # --- PLOT ELEMENT 1: The Raw Data (Raster Plot) ---
    # Plot tiny ticks at the top for Correct, and at the bottom for Error
    correct_trials = df_all[df_all["is_correct"] == 1]["trial_num"]
    error_trials = df_all[df_all["is_correct"] == 0]["trial_num"]

    plt.scatter(
        correct_trials,
        np.ones(len(correct_trials)) * 1.05,
        marker="|",
        color="#5CB85C",
        alpha=0.5,
        label="Correct Trial",
    )
    plt.scatter(
        error_trials,
        np.ones(len(error_trials)) * -0.05,
        marker="|",
        color="#D9534F",
        alpha=0.5,
        label="Error Trial",
    )

    plt.plot(
        df_all["trial_num"],
        df_all["rolling_accuracy"],
        color="black",
        linewidth=2,
        label="Rolling Accuracy (Proxy)",
    )

    plt.axhline(
        y=0.60, color="blue", linestyle="--", linewidth=1.5, label="Proxy Threshold (0.70)"
    )

    plt.fill_between(
        df_all["trial_num"],
        -0.1,
        1.1,
        where=(df_all["engagement_state"] == 0),
        color="red",
        alpha=0.15,
        label="HMM: disengaged State",
        transform=plt.gca().get_xaxis_transform(),
    )

    plt.ylim(-0.1, 1.1)
    plt.xlim(0, len(df_all))
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("Fraction Correct", fontsize=12)
    plt.title("Session Trace: Rolling Accuracy Proxy vs. GLM-HMM States", fontsize=14, pad=15)

    plt.legend(loc="upper right", bbox_to_anchor=(1.22, 1), frameon=False)
    sns.despine()

    plt.tight_layout()
    plt.show()


def plot_cohort_overall_rt(df_rt, df_counts, ylim=None):
    ordered_contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
    palette = {"Expects Left": "#D9534F", "Expects Right": "#5CB85C"}

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    sns.lineplot(
        ax=axes[0],
        data=df_rt,
        x="contrast_ordinal",
        y="rt_sec",
        hue="Prior_Label",
        palette=palette,
        errorbar="se",
        marker="o",
        linewidth=2.5,
    )
    axes[0].set_title("Cohort Overall Reaction Time (Correct Trials)")
    axes[0].set_ylabel("Reaction Time (s)")
    axes[0].axvline(4, color="black", linestyle="--", alpha=0.3)

    sns.lineplot(
        ax=axes[1],
        data=df_counts,
        x="contrast_ordinal",
        y="n_trials",
        hue="Prior_Label",
        palette=palette,
        marker="o",
        linestyle="--",
        linewidth=1.5,
        legend=False,
    )
    axes[1].set_ylabel("Total Cohort Trials")
    axes[1].axvline(4, color="black", linestyle="--", alpha=0.3)
    axes[1].set_xticks(range(len(ordered_contrasts)))
    axes[1].set_xticklabels(ordered_contrasts, rotation=45)
    axes[1].set_xlabel("Signed Contrast")
    axes[1].set_xlim(-0.5, len(ordered_contrasts) - 0.5)

    if ylim:
        axes[0].set_ylim(ylim)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_cohort_rt_by_engagement(
    df_rt,
    df_counts,
    title="Reaction Time by Engagement",
    state_idx="proxy_state_label",
    drop_errors=False,
    ylim=None,
):
    ordered_contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
    palette = {"Expects Left": "#D9534F", "Expects Right": "#5CB85C"}
    if state_idx == "proxy_state_label":
        states = ["engaged", "disengaged"]
    elif state_idx == "engagement_state":
        states = df_rt[state_idx].unique()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(states),
        figsize=(6 * len(states), 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(title, y=1.02, fontsize=14)

    for col_idx, state in enumerate(states):
        rt_data = df_rt[df_rt[state_idx] == state]
        count_data = df_counts[df_counts[state_idx] == state]

        sns.lineplot(
            ax=axes[0, col_idx],
            data=rt_data,
            x="contrast_ordinal",
            y="rt_sec",
            hue="Prior_Label",
            palette=palette,
            errorbar=None if drop_errors else "se",
            marker="o",
            linewidth=2.5,
            legend=(col_idx == 0),
        )
        axes[0, col_idx].set_title(f"{state} State")
        axes[0, col_idx].set_ylabel("Reaction Time (s)" if col_idx == 0 else "")
        axes[0, col_idx].axvline(4, color="black", linestyle="--", alpha=0.3)

        sns.lineplot(
            ax=axes[1, col_idx],
            data=count_data,
            x="contrast_ordinal",
            y="n_trials",
            hue="Prior_Label",
            palette=palette,
            marker="o",
            linestyle="--",
            linewidth=1.5,
            legend=False,
        )
        axes[1, col_idx].set_ylabel("Total Cohort Trials" if col_idx == 0 else "")
        axes[1, col_idx].axvline(4, color="black", linestyle="--", alpha=0.3)
        axes[1, col_idx].set_xticks(range(len(ordered_contrasts)))
        axes[1, col_idx].set_xticklabels(ordered_contrasts, rotation=45)
        axes[1, col_idx].set_xlabel("Signed Contrast")
        axes[1, col_idx].set_xlim(-0.5, len(ordered_contrasts) - 0.5)

    if ylim:
        axes[0, 0].set_ylim(ylim)
        axes[0, 1].set_ylim(ylim)

    sns.despine()
    plt.tight_layout()
    plt.show()


def run_all_behavioral_lmms(df_cohort, state_column="proxy_state_label"):

    print(f"========== INITIALIZING MODELS USING: {state_column} ==========\n")

    df = df_cohort.copy()

    df["Congruence_Label"] = df["Congruence_Label"].astype(str)
    df[state_column] = df[state_column].astype(str)

    if "Accuracy_Label" not in df.columns:
        df["Accuracy_Label"] = (
            df["feedbackType"].map({1.0: "Correct", -1.0: "Incorrect"}).astype(str)
        )

    df_correct = df[df["feedbackType"] == 1.0].copy()

    results_dict = {}

    print("============Correct Trials===========")
    formula_1 = (
        f"rt_sec ~ C(Congruence_Label, Treatment('Congruent')) + "
        f"C({state_column}, Treatment('engaged')) + "
        f"absolute_contrast"
    )
    mdl_1 = smf.mixedlm(formula_1, data=df_correct, groups=df_correct["animal_id"])
    res_1 = mdl_1.fit()
    print(res_1.summary())
    results_dict["main_effects"] = res_1
    print("\n" + "=" * 80 + "\n")

    print("============Correct Trials Interaction===========")

    formula_2 = (
        f"rt_sec ~ C(Congruence_Label, Treatment('Congruent')) * "
        f"C({state_column}, Treatment('engaged')) + "
        f"absolute_contrast"
    )
    mdl_2 = smf.mixedlm(formula_2, data=df_correct, groups=df_correct["animal_id"])
    res_2 = mdl_2.fit()
    print(res_2.summary())
    results_dict["2_way_interaction"] = res_2
    print("\n" + "=" * 80 + "\n")

    print("============All Trials===========")
    formula_3 = (
        f"rt_sec ~ C(Congruence_Label, Treatment('Congruent')) * "
        f"C({state_column}, Treatment('engaged')) * "
        f"C(Accuracy_Label, Treatment('Correct')) + "
        f"absolute_contrast"
    )
    mdl_3 = smf.mixedlm(formula_3, data=df, groups=df["animal_id"])
    res_3 = mdl_3.fit()
    print(res_3.summary())
    results_dict["3_way_interaction"] = res_3

    return results_dict


def get_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def clean_lmm_results(results_dict):

    cleaned_tables = []

    for model_name, result in results_dict.items():

        df_summary = pd.DataFrame(
            {
                "Model": model_name.replace("_", " ").title(),
                "Effect (ms)": (result.params * 1000).round(1),
                "StdErr (ms)": (result.bse * 1000).round(1),
                "p-value": result.pvalues,
                "z-score": result.tvalues.round(2),
            }
        )
        df_summary["Sig."] = df_summary["p-value"].apply(get_stars)
        df_summary["p-value"] = df_summary["p-value"].round(4)
        clean_index = df_summary.index.to_series()
        clean_index = clean_index.str.replace(r"C\([^\]]+?\)\[T\.([^\]]+)\]", r"\1", regex=True)
        clean_index = clean_index.str.replace(":", " x ")
        clean_index = clean_index.replace("absolute_contrast", "Contrast (0 to 1)")
        df_summary.index = clean_index

        if "Group Var" in df_summary.index:
            df_summary = df_summary.drop("Group Var")

        cleaned_tables.append(df_summary)
    master_df = pd.concat(cleaned_tables)

    master_df = master_df[["Model", "Effect (ms)", "StdErr (ms)", "z-score", "p-value", "Sig."]]

    return master_df


def load_trials_sebi_engagement(
    engagement_json,
    action_kernel_files,
    eid,
    window_size=10,
    state_idx="engagement_state",
    plot=False,
):

    df = action_kernel_files[eid]
    engagement = engagement_json[eid]
    df["engagement"] = engagement

    df_alt = df.copy()
    df_alt = df_alt.reset_index(drop=True)
    df_alt["rt_sec"] = df_alt["response_times"] - df_alt["stimOn_times"]

    df_alt = df_alt[(df_alt["rt_sec"] >= 0.08) & (df_alt["rt_sec"] <= 2.0)].copy()
    # df_alt = df_alt[(df_alt["rt_sec"] >= 0.08)].copy()
    # df_alt["contrastRight"] = df_alt["contrastRight"].fillna(0)
    # df_alt["contrastLeft"] = df_alt["contrastLeft"].fillna(0)
    df_alt["stimulus_side"] = np.where(df_alt["contrastRight"].notna(), 1.0, -1.0)
    df_alt["absolute_contrast"] = df_alt[["contrastRight", "contrastLeft"]].max(axis=1)

    df_alt["expected_side"] = np.where(df_alt["prior"] > 0.5, -1.0, 1.0)
    df_alt["internal_congruent"] = df_alt["expected_side"] == df_alt["stimulus_side"]
    df_alt["Congruence_Label"] = df_alt["internal_congruent"].map(
        {True: "Congruent", False: "Incongruent"}
    )
    df_alt["signed_contrast"] = df_alt["absolute_contrast"] * df_alt["stimulus_side"]

    df_alt["Prior_Label"] = df_alt["expected_side"].map(
        {-1.0: "Expects Left", 1.0: "Expects Right"}
    )
    ordered_contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
    contrast_mapping = {val: i for i, val in enumerate(ordered_contrasts)}
    df_alt["contrast_ordinal"] = df_alt["signed_contrast"].map(contrast_mapping)

    df_alt["is_correct"] = (df_alt["feedbackType"] == 1.0).astype(int)
    engaged_threshold = 0.60

    df_alt["rolling_accuracy"] = (
        df_alt["is_correct"]
        .shift(1)
        .ewm(span=window_size, min_periods=1)
        .mean()
        # rolling(window=window_size, min_periods=1).mean()
    )

    df_alt["proxy_state_label"] = np.where(
        df_alt["rolling_accuracy"] >= engaged_threshold, "engaged", "disengaged"
    )
    df_alt["proxy_state_label"] = df_alt["proxy_state_label"].fillna("engaged")

    df_correct = df_alt[df_alt["feedbackType"] == 1.0].copy()
    df_incorrect = df_alt[df_alt["feedbackType"] == -1.0].copy()

    df_alt["is_correct"] = (df_alt["feedbackType"] == 1.0).astype(int)

    # state_idx = "proxy_state_label"
    if plot:
        plt.figure(figsize=(6, 5))

        sns.barplot(
            data=df_alt,
            x=state_idx,
            y="is_correct",
            # order=[0, 1],  # Forces engaged to be on the left
            # palette={"engaged": "#4C72B0", "disengaged": "#C44E52"},
            capsize=0.1,
            errorbar=("ci", 95),
        )

        plt.title("Proportion Correct by Engagement State", pad=15)
        plt.ylabel("Fraction Correct")
        plt.xlabel("")  # Leave blank for cleaner look
        plt.ylim(0, 1.05)  # Lock y-axis from 0 to 1
        plt.axhline(0.5, color="black", linestyle="--", alpha=0.5)  # Add chance-level line

        sns.despine()
        plt.tight_layout()
        plt.show()

    return df_correct, df_incorrect, df_alt


def run_all_behavioral_lmms_continuous(df_cohort, engagement_column="engagement_norm"):
    """
    Runs LMMs using a continuous engagement signal (0-1) instead of categories.
    """

    df = df_cohort.copy()

    df[engagement_column] = pd.to_numeric(df[engagement_column])

    avg_engagement = df[engagement_column].mean()
    df["engagement_centered"] = df[engagement_column] - avg_engagement

    print(f"Engagement Mean: {avg_engagement:.3f} (This is now your 0 point)\n")

    df["Congruence_Label"] = df["Congruence_Label"].astype(str)
    if "Accuracy_Label" not in df.columns:
        df["Accuracy_Label"] = (
            df["feedbackType"].map({1.0: "Correct", -1.0: "Incorrect"}).astype(str)
        )

    results_dict = {}

    # --- MODEL 1: Main Effects ---
    # Intercept = RT for a Congruent/Correct trial at AVERAGE engagement
    print("============ Correct Trials: Main Effects ============")
    formula_1 = (
        "rt_sec ~ C(Congruence_Label, Treatment('Congruent')) + "
        "engagement_centered + absolute_contrast"
    )
    mdl_1 = smf.mixedlm(
        formula_1,
        data=df[df["feedbackType"] == 1.0],
        groups=df[df["feedbackType"] == 1.0]["animal_id"],
    )
    res_1 = mdl_1.fit()
    print(res_1.summary())
    results_dict["main_effects"] = res_1

    # --- MODEL 2: Two-Way Interaction ---
    # Does the speed-penalty of an Incongruent trial change as engagement moves away from average?
    print("\n============ Correct Trials: 2-Way Interaction ============")
    formula_2 = (
        "rt_sec ~ C(Congruence_Label, Treatment('Congruent')) * "
        "engagement_centered + absolute_contrast"
    )
    mdl_2 = smf.mixedlm(
        formula_2,
        data=df[df["feedbackType"] == 1.0],
        groups=df[df["feedbackType"] == 1.0]["animal_id"],
    )
    res_2 = mdl_2.fit()
    print(res_2.summary())
    results_dict["2_way_interaction"] = res_2

    # --- MODEL 3: Three-Way Interaction ---
    print("\n============ All Trials: 3-Way Interaction ============")
    formula_3 = (
        "rt_sec ~ C(Congruence_Label, Treatment('Congruent')) * "
        "engagement_centered * C(Accuracy_Label, Treatment('Correct')) + "
        "absolute_contrast"
    )
    mdl_3 = smf.mixedlm(formula_3, data=df, groups=df["animal_id"])
    res_3 = mdl_3.fit()
    print(res_3.summary())
    results_dict["3_way_interaction"] = res_3

    return results_dict
