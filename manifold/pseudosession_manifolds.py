# load up pseudosessions
# compute distance travelled and timewise deltas
import pickle as pkl
import numpy as np
from glob import glob


def total_distance_travelled(condition):
    deltas = np.diff(condition, axis=1)
    delta_sum = np.sum(np.abs(deltas), axis=1)
    return delta_sum


def test_for_entire_region(session, n_pseudosession=200):

    distance_matrix = []
    ca_matrix = []
    cb_matrix = []
    for pseudo_idx in range(n_pseudosession):
        n_conds = 2
        n_timepoints = 50
        stiched_session = session[pseudo_idx]

        cond_A_data = stiched_session[:, 0:n_timepoints]
        cond_B_data = stiched_session[:, n_timepoints:]

        distance = np.linalg.norm(cond_A_data - cond_B_data, axis=0)

        ca = total_distance_travelled(cond_A_data)
        cb = total_distance_travelled(cond_B_data)
        distance_matrix.append(distance)
        ca_matrix.append(ca)
        cb_matrix.append(cb)

    return distance_matrix, ca_matrix, cb_matrix


def process_eids(pickle_dump):

    total_distance_pseudosession = []
    distance_matrix = []
    for eids in pickle_dump.keys():
        dist, ca, cb = test_for_entire_region(pickle_dump[eids])
        total_distance_pseudosession.append([ca, cb])
        distance_matrix.append(dist)
    return total_distance_pseudosession, distance_matrix


if __name__ == "__main__":
    folder = "./data/generated/manifold/pseudosessions/"

    filenames = glob(folder + "*.pkl")

    for fname in filenames:
        try:
            with open(fname, "rb") as f:
                pickle_dump = pkl.load(f)
            total_distance_pseudosession, distance_matrix = process_eids(pickle_dump)
            region_name = fname.split("_pseudosession.pkl")[0].split("_")[-1]
            region_pickle = {}
            region_pickle["total_distance_pseudosession"] = total_distance_pseudosession
            region_pickle["distance_matrix"] = distance_matrix
            with open(f"pseudosession_{region_name}_metrics.pkl", "wb") as f:
                pkl.dump(region_pickle, f)
            print(f"Saved")
        except Exception as e:
            print(e)
