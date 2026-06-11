# load up pseudosessions
# compute distance travelled and timewise deltas
import pickle as pkl
import numpy as np
from glob import glob
from scipy.spatial.distance import cdist, cosine


def analyze_stitched_manifold(region_data, n_timepoints=50, soft_norm_factor=5.0):
    """
    Loads session PSTHs, stitches them into a supersession, normalizes,
    and computes state-space geometry and trajectory metrics.
    """

    neuron_blocks = [psth_matrix for session_id, psth_matrix in region_data.items()]
    supersession = np.vstack(neuron_blocks)

    # neuron_ranges = np.ptp(supersession, axis=1, keepdims=True)
    # supersession_norm = supersession / (neuron_ranges + soft_norm_factor)

    cond_correct = supersession[:, :n_timepoints]
    cond_incorrect = supersession[:, n_timepoints:]
    centroid_correct = np.mean(cond_correct, axis=1)
    centroid_incorrect = np.mean(cond_incorrect, axis=1)
    real_similarity = 1 - cosine(centroid_correct, centroid_incorrect)

    len_correct = np.sum(np.linalg.norm(np.diff(cond_correct, axis=1), axis=0))
    len_incorrect = np.sum(np.linalg.norm(np.diff(cond_incorrect, axis=1), axis=0))

    path_length_difference = len_correct - len_incorrect

    return {
        "N_total_neurons": supersession.shape[0],
        "centroid_similarity": real_similarity,
        "path_length_correct": len_correct,
        "path_length_incorrect": len_incorrect,
        "path_length_diff": path_length_difference,
    }


def compute_null_distributions(pseudosession_data, n_timepoints=50, soft_norm_factor=5.0):
    """
    Iterates through pseudosessions, stitches them across eids,
    and computes the null distributions for manifold metrics.
    """
    eids = list(pseudosession_data.keys())

    # this is normally 200
    n_pseudosessions = len(pseudosession_data[eids[0]])

    null_distributions = {
        "centroid_similarity": np.zeros(n_pseudosessions),
        "path_length_correct": np.zeros(n_pseudosessions),
        "path_length_incorrect": np.zeros(n_pseudosessions),
        "path_length_diff": np.zeros(n_pseudosessions),
    }

    for p_id in range(n_pseudosessions):

        pseudo_region_data = {eid: pseudosession_data[eid][p_id] for eid in eids}

        metrics = analyze_stitched_manifold(
            pseudo_region_data, n_timepoints=n_timepoints, soft_norm_factor=soft_norm_factor
        )

        null_distributions["centroid_similarity"][p_id] = metrics["centroid_similarity"]
        null_distributions["path_length_correct"][p_id] = metrics["path_length_correct"]
        null_distributions["path_length_incorrect"][p_id] = metrics["path_length_incorrect"]
        null_distributions["path_length_diff"][p_id] = metrics["path_length_diff"]

    return null_distributions


if __name__ == "__main__":
    folder = "./data/generated/manifold/pseudosessions/"

    filenames = glob(folder + "*.pkl")

    for fname in filenames:
        try:
            with open(fname, "rb") as f:
                pickle_dump = pkl.load(f)
            null_distirbutions = compute_null_distributions(pickle_dump)
            region_name = fname.split("_pseudosession.pkl")[0].split("_")[-1]
            with open(f"{folder}/pseudosession_{region_name}_metrics_updated.pkl", "wb") as f:
                pkl.dump(null_distirbutions, f)
            print(f"Saved")
        except Exception as e:
            print(e)
