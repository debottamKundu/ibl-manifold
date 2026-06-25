import traceback

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle as pkl
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from glob import glob
import warnings

warnings.filterwarnings("ignore")

def get_trajectory_data_real_session(data_true, data_pseudo):

    stiched_session = []
    for k in data_true.keys():
        if k in data_pseudo.keys():
            stiched_session.append(data_true[k])
    stiched_session = np.concatenate(stiched_session)
     
    pca = PCA(n_components=3)
    pca_session = pca.fit_transform(stiched_session.T)
    # print(pca_session.shape)
    n_timepoints = 50
    n_conditions = 4
    cond_A_data = pca_session[0:50,:]
    cond_B_data = pca_session[50:100,:]
    cond_C_data = pca_session[100:150,:]
    cond_D_data = pca_session[150:,:]
    # print(f'{region}: {np.sum(pca.explained_variance_ratio_)}')
    #plot_trajectories_multiple(cond_A_data, cond_B_data, cond_C_data, cond_D_data, region)
    return np.asarray([cond_A_data, cond_B_data, cond_C_data, cond_D_data]), pca

def get_trajectory_data_pseudosession(data_true, data_pseudo, pca_object, n_pseudo=200): # i think
    stiched_session = []
    for idx in range(n_pseudo):
        temp = []
        for k in data_pseudo.keys():
            if k in data_true.keys():
                temp.append(data_pseudo[k][idx])
            
        temp = np.concatenate(temp)
        temp = np.nan_to_num(temp)
        if pca_object is None:
            pca = PCA(n_components=3)
            pca_session = pca.fit_transform(temp.T)
        else:
            pca_session = pca_object.transform(temp.T)
        cond_A_data = pca_session[0:50,:]
        cond_B_data = pca_session[50:100,:]
        cond_C_data = pca_session[100:150,:]
        cond_D_data = pca_session[150:,:]
        stiched_session.append([cond_A_data, cond_B_data, cond_C_data, cond_D_data])
    return np.asarray(stiched_session)

def compute_avg_distance(traj1, traj2):
    return np.mean(np.linalg.norm(traj1 - traj2, axis=1))

def compute_centroid_difference(traj1, traj2):
    centroid1 = np.mean(traj1, axis=0)
    centroid2 = np.mean(traj2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)

def compute_pvalforhypothesis(real_data, pseudosession_data, plot=False, n_pseudosessions=200):
    real_dist_incongruent = compute_centroid_difference(traj1=real_data[0,:], traj2=real_data[1,:])
    real_dist_correct = compute_centroid_difference(traj1=real_data[0,:], traj2=real_data[2,:])

    null_distance_incongruent = []
    null_distance_correct = []

    for idx in range(n_pseudosessions):
        null_distance_incongruent.append(compute_centroid_difference(pseudosession_data[idx, 0, :], pseudosession_data[idx, 1, :]))
        null_distance_correct.append(compute_centroid_difference(pseudosession_data[idx, 0, :], pseudosession_data[idx, 2, :]))

    null_distance_incongruent = np.array(null_distance_incongruent)
    null_distance_correct = np.array(null_distance_correct)

    # p_value_A = np.mean(null_distance_incongruent >= real_dist_incongruent)
    # p_value_B = np.mean(null_distance_correct <= real_dist_correct)

    # if plot:
    #     fig,ax = plt.subplots(figsize=(12,3), ncols=2)

    #     sns.histplot(null_distance_incongruent, ax=ax[0])
    #     ax[0].axvline(real_dist_incongruent)
    #     ax[0].set_title(f'IC: R vs W, {p_value_A:.2f}')


    #     sns.histplot(null_distance_correct, ax=ax[1])
    #     ax[1].axvline(real_dist_correct)
    #     ax[1].set_title(f'IC v C, {p_value_B:.2f}')

    return real_dist_correct, real_dist_incongruent, null_distance_correct, null_distance_incongruent
def get_aligned_trajectory_data(data_true, data_pseudo, n_pseudo=200, output_type='pca'):

    shared_keys = sorted([k for k in data_true.keys() if k in data_pseudo.keys()]) 
    valid_keys = [k for k in shared_keys if data_true[k].shape[0] == data_pseudo[k][0].shape[0]]
    

    true_list = [np.nan_to_num(data_true[k]) for k in valid_keys]
    stiched_true = np.concatenate(true_list, axis=0)
        
    if output_type == 'pca':
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(stiched_true.T)
    else:
        pca_data = stiched_true.T
        
    real_data = np.asarray([
        pca_data[0:50, :],
        pca_data[50:100, :],
        pca_data[100:150, :],
        pca_data[150:, :]
    ])
    
    all_pseudos = []
    for idx in range(n_pseudo):
        pseudo_list = [np.nan_to_num(data_pseudo[k][idx]) for k in valid_keys]
        stiched_pseudo = np.concatenate(pseudo_list, axis=0)
        
        if output_type == 'pca':
            proj_data = pca.transform(stiched_pseudo.T)
        else:
            proj_data = stiched_pseudo.T
            
        cond_data = np.asarray([
            proj_data[0:50, :],
            proj_data[50:100, :],
            proj_data[100:150, :],
            proj_data[150:, :]
        ])
        all_pseudos.append(cond_data)
        
    return real_data, np.asarray(all_pseudos)


if __name__ == "__main__":

    files = glob('./data/generated/manifold/true_results_congruence_included/*.pkl')
    dataframe = []
    for fname in files:

        try:
            rname = fname.rsplit("/")[-1].rsplit(".pkl")[0].rsplit("_")[1]

            pseudo_fname = f"./data/generated/manifold/1000pseudo/aggregated_{rname}_pseudosession.pkl"
            d_pseudo = pkl.load(open(pseudo_fname,'rb'))
            d_true = pkl.load(open(fname,'rb'))
            
            
            real_data, null_data = get_aligned_trajectory_data(d_true, d_pseudo, output_type='pca')
            # check_neuron_mismatches(d_true, d_pseudo)
            
            real_correct, real_incongruent, null_correct, null_incongruent = compute_pvalforhypothesis(real_data, null_data)
            dataframe.append(
                {
                    'region':rname,
                    "correct_congruent_incongruent":real_correct,
                    "incongruent_condition_distance":real_incongruent,
                    "null_corrects":null_correct,
                    "null_incongruents":null_incongruent
                }
            )
            # if pvala<0.05:
            #     fig = plt.figure(figsize=(12, 6))
            #     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            #     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            #     plot_trajectories_on_ax(ax1, real_data[0,:], real_data[1,:], real_data[2,:], real_data[3,:], f'true-{rname}')
            #     A,B,C,D = null_data[0, 0,:], null_data[0, 1,:], null_data[0, 2,:], null_data[0, 3,:]
            #     plot_trajectories_on_ax(ax2, A, B, C, D, "Null Data")
            #     ax1.legend()
            #     plt.tight_layout()
            #     plt.show()
        except Exception as e:
            # print(e, rname)
            print(rname, fname, pseudo_fname)
            # traceback.print_exc()
            break

    dataframe = pd.DataFrame(dataframe)
    dataframe.to_parquet('./data/generated/manifold/summary.pqt')