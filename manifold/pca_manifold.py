import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import pearsonr


def unstack_pseudosession(stacked_matrix, cond_names, n_timebins):
    """
    Chops the horizontally stacked PSTH matrix back into a dictionary.
    Returns: dict[cond_name] = matrix of shape (n_neurons, n_timebins)
    """
    n_neurons = stacked_matrix.shape[0]
    unstacked = {}

    for i, cond in enumerate(cond_names):
        start_idx = i * n_timebins
        end_idx = (i + 1) * n_timebins
        unstacked[cond] = stacked_matrix[:, start_idx:end_idx]

    return unstacked


def plot_state_space_collapse(unstacked_dict, region="RegionName"):
    """
    Projects Engaged and Disengaged trajectories into the Engaged PCA space.
    Assumes standard split (include_history=False).
    """
    eng_cong = unstacked_dict["Eng_Cong_Corr"]
    eng_incong = unstacked_dict["Eng_Incong_Corr"]

    dis_cong = unstacked_dict["Dis_Cong_Corr"]
    dis_incong = unstacked_dict["Dis_Incong_Corr"]

    engaged_data = np.hstack([eng_cong, eng_incong]).T
    # mean_activity = np.mean(engaged_data, axis=0)

    pca = PCA(n_components=3)
    # pca.fit(engaged_data - mean_activity)
    pca.fit(engaged_data)

    def project(psth):
        # return pca.transform((psth.T - mean_activity))
        return pca.transform(psth.T)

    traj_eng_cong = project(eng_cong)
    traj_eng_incong = project(eng_incong)
    traj_dis_cong = project(dis_cong)
    traj_dis_incong = project(dis_incong)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle(f"{region} Quiescent Manifold: State-Space Collapse", fontsize=16)

    c_cong = "#2CA02C"
    c_incong = "#D62728"

    def plot_traj(ax, traj, color, label):
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.5, label=label)

        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=50, marker="o")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=100, marker="x")

    plot_traj(axes[0], traj_eng_cong, c_cong, "Congruent")
    plot_traj(axes[0], traj_eng_incong, c_incong, "Incongruent")
    axes[0].set_title("Engaged State")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].legend()

    plot_traj(axes[1], traj_dis_cong, c_cong, "Congruent")
    plot_traj(axes[1], traj_dis_incong, c_incong, "Incongruent")
    axes[1].set_title("Disengaged State")
    axes[1].set_xlabel("PC 1")

    sns.despine()
    plt.tight_layout()
    plt.show()
