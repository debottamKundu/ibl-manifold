from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, Ridge, RidgeCV
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import seaborn as sns


def ideal_rsa_matrices():
    """
    Constructs 8x8 Model RDMs (Representational Dissimilarity Matrices).
    Returns a dictionary of flattened model vectors (upper triangle) to use as predictors.

    Indices (8 conditions):
    0: L_Cong_Corr  (Stim L, Block L, Move L)
    1: L_Cong_Err   (Stim L, Block L, Move R)
    2: L_Incong_Corr   (Stim L, Block R, Move L)
    3: L_Incong_Err    (Stim L, Block R, Move R)
    4: R_Cong_Corr  (Stim R, Block R, Move R)
    5: R_Cong_Err   (Stim R, Block R, Move L)
    6: R_Incong_Corr   (Stim R, Block L, Move R)
    7: R_Incong_Err    (Stim R, Block L, Move L)
    """

    # 1. Initialize empty matrices
    n_conds = 8
    models = {
        "Choice": np.zeros((n_conds, n_conds)),
        "Prior": np.zeros((n_conds, n_conds)),
        "Congruence": np.zeros((n_conds, n_conds)),
        "Outcome": np.zeros((n_conds, n_conds)),
        "Stimulus": np.zeros((n_conds, n_conds)),
    }

    idx_move_L = [0, 2, 5, 7]
    idx_move_R = [1, 3, 4, 6]

    idx_block_L = [0, 1, 6, 7]
    idx_block_R = [2, 3, 4, 5]

    idx_corr = [0, 2, 4, 6]
    idx_err = [1, 3, 5, 7]

    idx_stim_L = [0, 1, 2, 3]
    idx_stim_R = [4, 5, 6, 7]

    idx_congruent = [0, 1, 4, 5]
    idx_incongruent = [2, 3, 6, 7]

    for r in range(n_conds):
        for c in range(n_conds):
            # Choice Model
            if (r in idx_move_L) != (c in idx_move_L):
                models["Choice"][r, c] = 1

            # Prior Model
            if (r in idx_block_L) != (c in idx_block_L):
                models["Prior"][r, c] = 1

            # Outcome Model
            if (r in idx_corr) != (c in idx_corr):
                models["Outcome"][r, c] = 1

            # Stimulus Model
            if (r in idx_stim_L) != (c in idx_stim_L):
                models["Stimulus"][r, c] = 1

            # Congruence Model
            if (r in idx_congruent) != (c in idx_congruent):
                models["Congruence"][r, c] = 1

    triu_indices = np.triu_indices(n_conds, k=1)

    predictors = {}
    for name, matrix in models.items():
        predictors[name] = matrix[triu_indices]

    return predictors, list(models.keys()), models


def simpler_rsa_matrices():
    """
    Generate 4x4 Model RDMs

    """
    n_conds = 4
    models = {
        "Choice": np.zeros((n_conds, n_conds)),
        "Prior": np.zeros((n_conds, n_conds)),
        "Congruence": np.zeros((n_conds, n_conds)),
    }

    # conditions:
    # L-cong, L-incong, R-cong, R-incong
    # new condition order:
    # dict_keys(["L_Cong_Corr", "R_Incong_Corr", "R_Cong_Corr", "L_Incong_Corr"])

    # choice
    idx_choice_L = [0, 3]
    idx_choice_R = [2, 3]
    # prior
    idx_block_L = [0, 1]
    idx_block_R = [2, 3]
    # congruence
    idx_congruent = [0, 2]
    idx_incongruent = [1, 3]

    for r in range(n_conds):
        for c in range(n_conds):
            # choice model - same as stimulus model
            if (r in idx_choice_L) != (c in idx_choice_L):
                models["Choice"][r, c] = 1
            # prior model
            if (r in idx_block_L) != (c in idx_block_L):
                models["Prior"][r, c] = 1
            # congruence
            if (r in idx_congruent) != (c in idx_congruent):
                models["Congruence"][r, c] = 1

    triu_indices = np.triu_indices(n_conds, k=1)

    predictors = {}
    for name, matrix in models.items():
        predictors[name] = matrix[triu_indices]

    return predictors, list(models.keys()), models


def run_rsa_regression(
    accumulated_data,
    model_vectors=None,
    model_names=None,
    normalization=False,
    model_type=None,
    conditions=8,
):

    if model_vectors is None:
        model_vectors, model_names, _ = ideal_rsa_matrices()

    X = np.column_stack([model_vectors[name] for name in model_names])  # type: ignore

    results = {}

    for region in accumulated_data.keys():
        print(f"Processing {region}...")
        results[region] = {}
        epochs = ["Quiescent", "Stimulus", "Choice"]

        for epoch in epochs:
            session_matrices = accumulated_data[region][epoch]
            if not session_matrices:
                continue

            pop_matrix = np.vstack(session_matrices)

            if normalization:
                pop_matrix = zscore(pop_matrix, axis=1)
                pop_matrix = np.nan_to_num(pop_matrix)

            n_bins = int(pop_matrix.shape[1] / conditions)
            reshaped = np.transpose(
                pop_matrix.reshape(pop_matrix.shape[0], conditions, n_bins), (1, 2, 0)
            )
            betas_over_time = np.zeros((n_bins, len(model_names)))  # type: ignore
            r2_scores = np.zeros(n_bins)
            # residuals = np.zeros((n_bins, 28))

            if model_type == "Lasso":
                reg = LassoCV(cv=5, fit_intercept=True, positive=True)
            elif model_type == "Ridge":
                reg = RidgeCV(alphas=[0.1, 1.0, 10.0], fit_intercept=True)
            elif model_type == "NNLS":
                # Simple Linear Regression but forced positive
                reg = LinearRegression(fit_intercept=True, positive=True)
            else:
                reg = LinearRegression(fit_intercept=True)

            for t in range(n_bins):
                trajectories = reshaped[:, t, :]

                y = pdist(trajectories, "euclidean")
                try:
                    reg.fit(X, y)
                    betas_over_time[t, :] = reg.coef_
                    r2_scores[t] = reg.score(X, y)

                    y_pred = reg.predict(X)
                    # residuals[t, :] = y - y_pred
                except Exception as e:
                    # Handle rare convergence errors
                    betas_over_time[t, :] = 0
                    r2_scores[t] = 0

            results[region][epoch] = {
                "betas": betas_over_time,
                "r2": r2_scores,
                # "residuals": residuals,
            }

    return results


def plot_rsa_dynamics(results, region, model_names, bin_size_s=0.01):
    """
    Plots time-resolved Beta weights and R2 for a specific region across epochs.

    Parameters:
    - results: Dictionary output from run_rsa_regression
    - region: String name of the region to plot
    - model_names: List of strings (e.g. ['Choice', 'Prior', ...])
    - bin_size_s: Time bin size in seconds (for x-axis scaling)
    """

    if region not in results:
        print(f"Region {region} not found in results.")
        return

    epochs = ["Quiescent", "Stimulus", "Choice"]
    region_data = results[region]

    # 1. Setup Figure: 2 Rows (Betas, R2), 3 Columns (Epochs)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex="col")
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    fig.suptitle(f"{region}: RSA Model Dynamics", fontsize=16, y=0.98)

    # Define consistent colors
    colors = sns.color_palette("bright", len(model_names))
    color_map = dict(zip(model_names, colors))

    # 2. Determine Y-Axis Limits for Betas (Global scaling for comparison)
    all_betas = []
    for ep in epochs:
        if ep in region_data:
            all_betas.append(region_data[ep]["betas"])

    if not all_betas:
        return

    flat_betas = np.concatenate(all_betas)
    y_min, y_max = np.min(flat_betas), np.max(flat_betas)
    # Add 10% padding
    # y_pad = (y_max - y_min) * 0.1
    # ylim = (y_min - y_pad, y_max + y_pad)

    # 3. Plot Loops
    for i, epoch in enumerate(epochs):
        ax_beta = axes[i]
        if epoch not in region_data:
            ax_beta.text(0.5, 0.5, "No Data", ha="center")
            continue

        data = region_data[epoch]
        betas = data["betas"]  # (n_bins, n_models)

        # Create Time Axis (seconds)
        # Adjust start time based on alignment if known (approx defaults used here)
        n_bins = betas.shape[0]
        if epoch == "Quiescent":
            start_t = -0.5
        elif epoch == "Stimulus":
            start_t = 0.0
        elif epoch == "Choice":
            start_t = -0.1

        t = np.linspace(start_t, start_t + n_bins * bin_size_s, n_bins)

        for k, name in enumerate(model_names):
            ax_beta.plot(t, betas[:, k], label=name, color=color_map[name], lw=2)

        ax_beta.set_title(epoch, fontsize=14, fontweight="bold")
        ax_beta.axhline(0, color="black", linestyle=":", alpha=0.5)
        # ax_beta.set_ylim(ylim)
        ax_beta.grid(True, alpha=0.3)

        if i == 0:
            ax_beta.set_ylabel("Kernel Strength (Beta)", fontsize=12)
            ax_beta.legend(loc="upper left", fontsize="small", framealpha=0.9)
    sns.despine()
    plt.show()


KERNEL_COLORS = {
    "Choice": "#d95f02",  # Orange
    "Prior": "#1b9e77",  # Green
    "Congruence": "#7570b3",  # Purple
    "Outcome": "#e7298a",  # Pink
    "Stimulus": "#66a61e",  # Olive
}


def plot_rsa_summary_bars(results, model_names):
    """
    Plots the mean Beta weights for each kernel, separated by Epoch.
    Includes SEM error bars representing variability over time within the epoch.

    Args:
        results (dict): Output from run_rsa_regression
        model_names (list): List of model names corresponding to the columns in 'betas'
    """

    epochs_ordered = ["Quiescent", "Stimulus", "Choice"]
    records = []

    for region, region_data in results.items():
        for epoch in epochs_ordered:
            if epoch not in region_data:
                continue

            betas_over_time = region_data[epoch]["betas"]

            for i, name in enumerate(model_names):
                kernel_betas = betas_over_time[:, i]
                for val in kernel_betas:
                    records.append({"Region": region, "Epoch": epoch, "Kernel": name, "Beta": val})

    df = pd.DataFrame(records)

    if df.empty:
        print("No data found to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 18))
    fig.suptitle("RSA Model Weights by Region and Epoch", fontsize=16, y=1.05)

    for i, epoch in enumerate(epochs_ordered):
        ax = axes[i]
        epoch_data = df[df["Epoch"] == epoch]

        if epoch_data.empty:
            ax.set_title(f"{epoch} (No Data)")
            ax.axis("off")
            continue

        sns.barplot(
            data=epoch_data,
            x="Region",
            y="Beta",
            hue="Kernel",
            ax=ax,
            palette=KERNEL_COLORS,  # 2. Pass the dictionary directly here!
            edgecolor="black",
            errorbar="se",
            capsize=0.1,
        )

        ax.set_title(epoch, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=90)

        if i == 0:
            ax.set_ylabel("Mean Beta Weight")
        else:
            ax.set_ylabel("")

        # Put legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Kernel", frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.show()

    return df


import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV


def run_rsa_regression_with_reward(
    accumulated_data,
    model_vectors=None,
    model_names=None,
    normalization=False,
    model_type=None,
    conditions=8,
    reward_dict=None,
    condition_keys=None,
    switch=False,
):

    if model_vectors is None:
        model_vectors, model_names, _ = ideal_rsa_matrices()

    X_base = np.column_stack([model_vectors[name] for name in model_names])  # type: ignore

    results = {}

    if reward_dict is not None and condition_keys is not None:
        global_rewards = np.zeros(conditions)
        all_animals = list(reward_dict.keys())

        if switch:
            k = "avg_number_of_correct_preceding_n"
        else:
            k = "avg_proportion_correct_preceding_n"
        for i, cond in enumerate(condition_keys):
            vals = [
                reward_dict[anim][cond][k]
                for anim in all_animals
                if anim in reward_dict and cond in reward_dict[anim]
            ]
            global_rewards[i] = np.nanmean(vals) if vals else 0
        global_reward_rdm = pdist(global_rewards.reshape(-1, 1), metric="euclidean")
    else:
        global_reward_rdm = np.zeros((conditions, conditions))

    for region in accumulated_data.keys():
        print(f"Processing {region}...")
        results[region] = {}

        X_region = X_base.copy()
        current_model_names = list(model_names)  # type: ignore

        # if reward_dict is not None and region_animals is not None and condition_keys is not None:

        #     animals_in_region = region_animals.get(region, [])

        #     region_rewards = np.zeros(conditions)
        #     for i, cond in enumerate(condition_keys):

        #         vals = [
        #             reward_dict[anim][cond]["avg_number_of_correct_preceding_n"]
        #             for anim in animals_in_region
        #             if anim in reward_dict and cond in reward_dict[anim]
        #         ]

        #         # region_rewards[i] = np.nanmean(vals) if vals else 0
        #         if not vals:
        #             print(f"  WARNING: No data for {cond} in {region}!")

        #         region_rewards[i] = (
        #             np.nanmean(vals) if vals else np.nan
        #         )  # Use NaN instead of 0 to spot it

        #     print(
        #         f"Region: {region} | N animals: {len(animals_in_region)} | Rewards: {np.round(region_rewards, 2)}"
        #     )

        #     reward_rdm = pdist(region_rewards.reshape(-1, 1), metric="euclidean")

        X_region = np.column_stack((X_region, global_reward_rdm))
        current_model_names.append("Previous-Outcome")

        X_region = zscore(X_region, axis=0)
        # X_region_standardized = np.nan_to_num(
        #     X_region_standardized
        # )  # Catch constant vectors if any

        epochs = ["Quiescent", "Stimulus", "Choice"]

        for epoch in epochs:
            session_matrices = accumulated_data[region][epoch]
            if not session_matrices:
                continue

            pop_matrix = np.vstack(session_matrices)

            if normalization:
                pop_matrix = zscore(pop_matrix, axis=1)
                pop_matrix = np.nan_to_num(pop_matrix)  # fixed variable name here

            n_bins = int(pop_matrix.shape[1] / conditions)
            reshaped = np.transpose(
                pop_matrix.reshape(pop_matrix.shape[0], conditions, n_bins), (1, 2, 0)
            )

            # Initialize betas array using the updated number of models
            betas_over_time = np.zeros((n_bins, len(current_model_names)))
            r2_scores = np.zeros(n_bins)

            if model_type == "Lasso":
                reg = LassoCV(cv=5, fit_intercept=True, positive=True)
            elif model_type == "Ridge":
                reg = RidgeCV(alphas=[0.1, 1.0, 10.0], fit_intercept=True)
            elif model_type == "NNLS":
                reg = LinearRegression(fit_intercept=True, positive=True)
            else:
                reg = LinearRegression(fit_intercept=True)

            for t in range(n_bins):
                trajectories = reshaped[:, t, :]

                y = pdist(trajectories, "euclidean")

                try:
                    # Fit using the Region-Specific X matrix
                    reg.fit(X_region, y)
                    betas_over_time[t, :] = reg.coef_
                    r2_scores[t] = reg.score(X_region, y)
                except Exception as e:
                    betas_over_time[t, :] = 0
                    r2_scores[t] = 0

            results[region][epoch] = {
                "betas": betas_over_time,
                "r2": r2_scores,
                "model_names": current_model_names,
                "zscored_xregion": X_region,
                "global_reward": global_reward_rdm,
            }

    return results


def dynamic_rsa_matrices(conditions):
    """
    Constructs Model RDMs dynamically based on condition attributes.

    Parameters:
    conditions (list of dicts): A list where each dictionary contains the
                                features of a condition in the current order.

    Returns:
    tuple: (predictors dict, list of model names, full models dict)
    """
    # example
    # current_condition_order =
    # {"Name": "L_Cong_Corr",   "Stim": "L", "Block": "L", "Move": "L", "Cong": "Cong",   "Out": "Corr"},
    # {"Name": "L_Cong_Err",    "Stim": "L", "Block": "L", "Move": "R", "Cong": "Cong",   "Out": "Err"},
    # {"Name": "L_Incong_Corr", "Stim": "L", "Block": "R", "Move": "L", "Cong": "Incong", "Out": "Corr"},
    # {"Name": "L_Incong_Err",  "Stim": "L", "Block": "R", "Move": "R", "Cong": "Incong", "Out": "Err"},
    # {"Name": "R_Cong_Corr",   "Stim": "R", "Block": "R", "Move": "R", "Cong": "Cong",   "Out": "Corr"},
    # {"Name": "R_Cong_Err",    "Stim": "R", "Block": "R", "Move": "L", "Cong": "Cong",   "Out": "Err"},
    # {"Name": "R_Incong_Corr", "Stim": "R", "Block": "L", "Move": "R", "Cong": "Incong", "Out": "Corr"},
    # {"Name": "R_Incong_Err",  "Stim": "R", "Block": "L", "Move": "L", "Cong": "Incong", "Out": "Err"}
    n_conds = len(conditions)

    # Initialize empty matrices
    models = {
        "Choice": np.zeros((n_conds, n_conds)),
        "Prior": np.zeros((n_conds, n_conds)),
        "Congruence": np.zeros((n_conds, n_conds)),
        "Outcome": np.zeros((n_conds, n_conds)),
        "Stimulus": np.zeros((n_conds, n_conds)),
    }

    for r in range(n_conds):
        for c in range(n_conds):

            # Choice Model (1 if different movement, 0 if same)
            if conditions[r]["Move"] != conditions[c]["Move"]:
                models["Choice"][r, c] = 1

            # Prior Model (1 if different block, 0 if same)
            if conditions[r]["Block"] != conditions[c]["Block"]:
                models["Prior"][r, c] = 1

            # Stimulus Model
            if conditions[r]["Stim"] != conditions[c]["Stim"]:
                models["Stimulus"][r, c] = 1

            # Congruence Model
            if conditions[r]["Cong"] != conditions[c]["Cong"]:
                models["Congruence"][r, c] = 1

            # Outcome Model
            if conditions[r]["Out"] != conditions[c]["Out"]:
                models["Outcome"][r, c] = 1

    # Flatten upper triangles for regression predictors
    triu_indices = np.triu_indices(n_conds, k=1)

    predictors = {}
    for name, matrix in models.items():
        predictors[name] = matrix[triu_indices]

    return predictors, list(models.keys()), models
