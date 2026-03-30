import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def unstack_pseudosession(stacked_matrix, cond_names, n_timebins):

    unstacked = {}
    for i, cond in enumerate(cond_names):
        start_idx = i * n_timebins
        end_idx = (i + 1) * n_timebins
        unstacked[cond] = stacked_matrix[:, start_idx:end_idx]

    return unstacked


def compute_separations(traj_eng_cong, traj_eng_incong, traj_dis_cong, traj_dis_incong):
    centroid_eng_cong = np.mean(traj_eng_cong, axis=0)
    centroid_eng_incong = np.mean(traj_eng_incong, axis=0)

    centroid_dis_cong = np.mean(traj_dis_cong, axis=0)
    centroid_dis_incong = np.mean(traj_dis_incong, axis=0)

    disengaged_separation = np.linalg.norm(centroid_dis_cong - centroid_dis_incong)
    engaged_separation = np.linalg.norm(centroid_eng_cong - centroid_eng_incong)

    return disengaged_separation, engaged_separation


def decode_congruence(traj_eng_cong, traj_eng_incong, traj_dis_cong, traj_dis_incong, n_splits=5):
    """
    Trains a linear SVM to classify Congruent vs. Incongruent trials.
    Returns the cross-validated accuracy for the Engaged and Disengaged states.

    Assumes input shapes are (n_trials, n_features) OR (n_trials, n_time, n_PCs).
    """

    # 1. Helper function to prepare data for a specific state
    def prepare_data(data_cong, data_incong):

        X_cong = data_cong.reshape(data_cong.shape[0], -1)
        X_incong = data_incong.reshape(data_incong.shape[0], -1)
        y_cong = np.zeros(X_cong.shape[0])
        y_incong = np.ones(X_incong.shape[0])
        X = np.vstack((X_cong, X_incong))
        y = np.concatenate((y_cong, y_incong))

        return X, y

    X_eng, y_eng = prepare_data(traj_eng_cong, traj_eng_incong)
    X_dis, y_dis = prepare_data(traj_dis_cong, traj_dis_incong)

    decoder = make_pipeline(StandardScaler(), LinearSVC(dual="auto", max_iter=10000))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_eng = np.mean(cross_val_score(decoder, X_eng, y_eng, cv=cv, scoring="accuracy"))
    acc_dis = np.mean(cross_val_score(decoder, X_dis, y_dis, cv=cv, scoring="accuracy"))

    std_eng = np.std(cross_val_score(decoder, X_eng, y_eng, cv=cv, scoring="accuracy"))
    std_dis = np.std(cross_val_score(decoder, X_dis, y_dis, cv=cv, scoring="accuracy"))

    print(f"--- Decoding Accuracy (Congruent vs. Incongruent) ---")
    print(f"Engaged State:    {acc_eng*100:.1f}%  (± {std_eng*100:.1f}%)")
    print(f"Disengaged State: {acc_dis*100:.1f}%  (± {std_dis*100:.1f}%)")
    print(f"Chance Level:     50.0%")

    return acc_eng, acc_dis


def plot_state_space_collapse(
    accumulated_data, cond_names, region="MOs", epoch="Quiescent", only_correct=True
):

    if region not in accumulated_data or epoch not in accumulated_data[region]:
        print(f"Data for {region} - {epoch} not found.")
        return

    session_matrices = accumulated_data[region][epoch]

    valid_matrices = [m for m in session_matrices if m is not None]
    if not valid_matrices:
        print(f"No valid session matrices for {region} - {epoch}.")
        return

    pseudosession_matrix = np.vstack(valid_matrices)
    print(f"Pseudosession built for {region}: {pseudosession_matrix.shape[0]} total neurons.")
    n_bins = int(
        pseudosession_matrix.shape[-1] / len(cond_names)
    )  # Divided by conditions conditions

    unstacked_dict = unstack_pseudosession(pseudosession_matrix, cond_names, n_bins)

    try:
        eng_cong = unstacked_dict["Eng_Cong_Corr"]
        eng_incong = unstacked_dict["Eng_Incong_Corr"]
        dis_cong = unstacked_dict["Dis_Cong_Corr"]
        dis_incong = unstacked_dict["Dis_Incong_Corr"]

        # incorrect
        eng_cong_err = unstacked_dict["Eng_Cong_Err"]
        eng_incong_err = unstacked_dict["Eng_Incong_Err"]
        dis_cong_err = unstacked_dict["Dis_Cong_Err"]
        dis_incong_err = unstacked_dict["Dis_Incong_Err"]

    except KeyError as e:
        print(f"Missing expected condition in data: {e}")
        return

    engaged_data = np.hstack([eng_cong, eng_incong]).T
    mean_activity = np.mean(engaged_data, axis=0)

    pca = PCA(n_components=3)
    pca.fit(engaged_data - mean_activity)
    print(pca.explained_variance_ratio_)

    def project(psth):
        return pca.transform((psth.T - mean_activity))

    traj_eng_cong = project(eng_cong)
    traj_eng_incong = project(eng_incong)
    traj_dis_cong = project(dis_cong)
    traj_dis_incong = project(dis_incong)

    # error projections
    traj_eng_cong_err = project(eng_cong_err)
    traj_eng_incong_err = project(eng_incong_err)
    traj_dis_cong_err = project(dis_cong_err)
    traj_dis_incong_err = project(dis_incong_err)

    if only_correct:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle(f"Manifold Trajectories: {region} {epoch} ", fontsize=16)

    c_cong = "#2CA02C"  # Green
    c_incong = "#D62728"  # Red

    def plot_traj(ax, traj, color, label, linestyle="-"):
        ax.plot(
            traj[:, 0], traj[:, 1], color=color, linewidth=2.5, linestyle=linestyle, label=label
        )
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=50, marker="o")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=100, marker="x")

    plot_traj(axes[0], traj_eng_cong, c_cong, "Congruent")
    plot_traj(axes[0], traj_eng_incong, c_incong, "Incongruent")

    # if "Eng_Incong_Err" in unstacked_dict:
    #     traj_eng_incong_err = project(unstacked_dict["Eng_Incong_Err"])
    #     plot_traj(
    #         axes[0],
    #         traj_eng_incong_err,
    #         c_incong,
    #         "Incongruent Error",
    #         linestyle=":",
    #     )

    axes[0].set_title("Engaged State - Correct")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].legend(loc="best")

    plot_traj(axes[1], traj_dis_cong, c_cong, "Congruent")
    plot_traj(axes[1], traj_dis_incong, c_incong, "Incongruent")
    axes[1].set_title("Disengaged State - Correct")
    axes[1].set_xlabel("PC 1")

    # incorrect
    if not only_correct:
        plot_traj(axes[2], traj_eng_cong_err, c_cong, "Congruent")
        plot_traj(axes[2], traj_eng_incong_err, c_incong, "Incongruent")

        axes[2].set_title("Engaged State - Incorrect")
        axes[2].set_xlabel("PC 1")
        axes[2].set_ylabel("PC 2")
        axes[2].legend(loc="best")

        plot_traj(axes[3], traj_dis_cong_err, c_cong, "Congruent")
        plot_traj(axes[3], traj_dis_incong_err, c_incong, "Incongruent")
        axes[3].set_title("Disengaged State - Incorrect")
        axes[3].set_xlabel("PC 1")

    sns.despine()
    plt.tight_layout()
    plt.show()

    disengaged_separation_correct, engaged_separation_correct = compute_separations(
        traj_eng_cong=traj_eng_cong,
        traj_eng_incong=traj_eng_incong,
        traj_dis_cong=traj_dis_cong,
        traj_dis_incong=traj_dis_incong,
    )
    disengaged_separation_incorrect, engaged_separation_incorrect = compute_separations(
        traj_eng_cong=traj_eng_cong_err,
        traj_eng_incong=traj_eng_incong_err,
        traj_dis_cong=traj_dis_cong_err,
        traj_dis_incong=traj_dis_incong_err,
    )
    # original space
    og_space_engaged, og_space_disengaged = compute_separations(
        traj_eng_cong=eng_cong.T,
        traj_eng_incong=eng_incong.T,
        traj_dis_cong=dis_cong.T,
        traj_dis_incong=dis_incong.T,
    )

    # plot
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)
    ax.flatten()
    ax[0].bar(
        np.arange(2),
        [
            engaged_separation_correct,
            disengaged_separation_correct,
            # engaged_separation_incorrect,
            # disengaged_separation_incorrect,
        ],
        edgecolor="k",
        color="#16da9c",
        alpha=0.75,
    )
    ax[0].set_xticks(np.arange(2))
    ax[0].set_xticklabels(["Engaged Correct", "Disengaged Correct"])
    ax[0].set_title("PCA space")

    ax[1].bar(
        np.arange(2),
        [
            og_space_engaged,
            og_space_disengaged,
            # engaged_separation_incorrect,
            # disengaged_separation_incorrect,
        ],
        edgecolor="k",
        color="#16da9c",
        alpha=0.75,
    )
    ax[1].set_xticks(np.arange(2))
    ax[1].set_xticklabels(["Engaged Correct", "Disengaged Correct"])
    ax[1].set_title("Complete space")

    fig.suptitle(f"Trajectory distances: {region} {epoch} ", fontsize=16)

    sns.despine()
    plt.tight_layout()
    plt.show()


def collapsed_rsa_matrices():
    """
    Generate 4x4 Model RDMs
    Indices (4 abstract conditions):
    0: Cong_Corr
    1: Cong_Err
    2: Incong_Corr
    3: Incong_Err
    """
    n_conds = 4
    models = {
        "Congruence": np.zeros((n_conds, n_conds)),
        "Accuracy": np.zeros((n_conds, n_conds)),
        "Followed_Prior": np.zeros((n_conds, n_conds)),
    }

    idx_congruent = [0, 1]
    idx_correct = [0, 2]
    idx_followed_prior = [0, 3]

    for r in range(n_conds):
        for c in range(n_conds):

            if (r in idx_congruent) != (c in idx_congruent):
                models["Congruence"][r, c] = 1

            if (r in idx_correct) != (c in idx_correct):
                models["Accuracy"][r, c] = 1

            if (r in idx_followed_prior) != (c in idx_followed_prior):
                models["Followed_Prior"][r, c] = 1

    triu_indices = np.triu_indices(n_conds, k=1)

    predictors = {}
    for name, matrix in models.items():
        predictors[name] = matrix[triu_indices]

    return predictors, list(models.keys()), models
