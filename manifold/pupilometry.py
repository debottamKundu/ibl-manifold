import tempfile
import pickle
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

from one.api import ONE
from scipy import stats
from scipy.special import logit, softmax
import os
from os.path import join
import pickle as pkl
from brainwidemap.bwm_loading import bwm_query, bwm_units, load_trials_and_mask, merge_probes

import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

from one.api import ONE
from brainbox.io.one import SessionLoader
from iblatlas.atlas import BrainRegions
from sklearn.metrics import balanced_accuracy_score

from brainwidemap.bwm_loading import (
    bwm_query,
    load_good_units,
    load_trials_and_mask,
    merge_probes,
    bwm_units,
)
import numpy as np
import pandas as pd
from scipy.stats import linregress, zscore
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import zscore




def check_session_significance(trials_df,pupil_df, window=(0,1), plot=False):

    trials_df = trials_df.reset_index(drop=True)
    mean_pupil_sizes = np.full(len(trials_df), np.nan)

    for index, trial in trials_df.iterrows():
        window_start = trial['stimOn_times'] + window[0] 
        window_end = trial['stimOn_times'] + window[1]
        
        time_mask = (pupil_df['times'] >= window_start) & (pupil_df['times'] <= window_end)
        
        window_data = pupil_df.loc[time_mask, 'pupilDiameter_raw']
        if not window_data.empty: # type: ignore
            mean_pupil_sizes[index] = np.nanmean(window_data) # type: ignore

    trials_df['pupil_mean'] = mean_pupil_sizes

    reg_df = trials_df.dropna(subset=['pupil_mean'])
    reg_df['abs'] = np.abs(np.nan_to_num(reg_df['contrastLeft']) - np.nan_to_num(reg_df['contrastRight']))
    pupil_fluctuations = reg_df['pupil_mean']
    
    result = linregress(reg_df['abs'], pupil_fluctuations)
    
    is_significant = result.pvalue < 0.05
    
    if plot:
        sns.regplot(
        x=reg_df['abs'], 
        y=reg_df['pupil_mean'], 
        scatter_kws={'alpha': 0.5}, 
        line_kws={'label': f"(r={result.rvalue:.2f}, p={result.pvalue:.3f})"} # type: ignore
        )
        plt.legend()
        sns.despine()

    return is_significant, result


def test_feedback_modulation(trials_df, pupil_df, plot=False):

    reward_responses = []
    error_responses = []
    
    valid_trials = trials_df.dropna(subset=['feedback_times', 'feedbackType'])
    
    for i, trial in valid_trials.iterrows():
        win_start = trial['feedback_times']-0.2
        win_end = trial['feedback_times'] + 1.5
        
        mask = (pupil_df['times'] >= win_start) & (pupil_df['times'] <= win_end)
        window_data = pupil_df.loc[mask, 'pupilDiameter_raw']
        
        if not window_data.empty:
            mean_response = np.nanmean(window_data)
            
            if trial['feedbackType'] == 1:
                reward_responses.append(mean_response)
            else:
                error_responses.append(mean_response)
                
    
    if len(reward_responses) > 5 and len(error_responses) > 5:
        t_stat, p_val = stats.ttest_ind(reward_responses, error_responses, nan_policy='omit')
        # if p_val < 0.05: # type: ignore
        #     print("Result: SIGNIFICANT feedback modulation.")
        # else:
        #     print("Result: NOT significant.")
            
        return p_val < 0.05 # type: ignore
    else:
        print("Insufficient data for statistical testing.")
        return False
    
def plot_uncorrected_feedback_psth(trials_df, pupil_df):

    pupil_df['session_zscore'] = zscore(pupil_df['pupilDiameter_raw'], nan_policy='omit')
    
    
    time_grid = np.arange(-0.25, 1.5, 0.016)
    aligned_trials = []
    
    valid_trials = trials_df.dropna(subset=['feedback_times', 'feedbackType'])
    
    for i, trial in valid_trials.iterrows():
        feedback_time = trial['feedback_times']
        fb_label = 'Reward' if trial['feedbackType'] == 1 else 'Error'
        
        win_start = feedback_time - 0.25
        win_end = feedback_time + 1.5
        
        
        mask = (pupil_df['times'] >= win_start) & (pupil_df['times'] <= win_end)
        t_raw = pupil_df.loc[mask, 'times'].values
        p_zscored = pupil_df.loc[mask, 'session_zscore'].values
        
        if len(t_raw) < 10 or np.isnan(p_zscored).all():
            continue
            
        t_shifted = t_raw - feedback_time 
        interpolator = interp1d(t_shifted, p_zscored, bounds_error=False, fill_value=np.nan)
        p_interp = interpolator(time_grid)
        
        trial_data = pd.DataFrame({
            'time': time_grid,
            'pupil_response': p_interp,
            'feedback': fb_label,
            'trial_id': i
        })
        aligned_trials.append(trial_data)
        
    all_data = pd.concat(aligned_trials, ignore_index=True)
    
    plt.figure(figsize=(9, 6))
    
    sns.lineplot(
        data=all_data, 
        x='time', 
        y='pupil_response', 
        hue='feedback', 
        palette={'Reward': '#2ca02c', 'Error': '#d62728'},
        linewidth=2
    )
    
    plt.axvline(0, color='black', linestyle='--', alpha=0.8, label='Feedback')
    
    plt.xlim(-0.30, 1.5)
    plt.xlabel('Time from Feedback (s)', fontsize=12)
    plt.ylabel('Pupil Size', fontsize=12)
    plt.title('Feedback modulation', fontsize=14)
    
    plt.legend()
    plt.tight_layout()
    sns.despine()
    
    return all_data

def feedback_pupilometry(one, eid, engagement_df, scalar_motivation, action_kernel):

    sess = SessionLoader(one, eid)
    sess.load_pupil()

    trials_df, mask = load_trials_and_mask(one, eid)
    animal_engagement = engagement_df[engagement_df['eid']==eid].reset_index(drop=True)
    stimulus_engage = scalar_motivation[eid]
    akernel_df = action_kernel[eid]


    trials_df = trials_df.merge(animal_engagement[['p_state1','p_state2','signed_contrast','rewarded']],left_index=True, right_index=True)
    trials_df['akernel_prior'] = akernel_df['prior']
    trials_df['pe_right'] = akernel_df['prediction_error_right']
    trials_df['motivation_scalar'] = stimulus_engage

    masked_trials_df = trials_df[mask].copy()
    masked_trials_df = masked_trials_df.dropna(subset=['feedback_times','feedbackType'])
    pupil_df = sess.pupil
    
    assert np.all(masked_trials_df['feedbackType']==masked_trials_df['rewarded'])

    all_pupil_sizes = []
    all_pupil_times = []

    # get feedback onset, 200ms before to 1.5 seconds after, no mean, entire time series.
    pupil_df['session_zscore'] = zscore(pupil_df['pupilDiameter_raw'], nan_policy='omit')
    for i, trial in masked_trials_df.iterrows():
        feedback_time = trial['feedback_times']
        win_start = feedback_time - 0.25
        win_end = feedback_time + 1.5
        
        
        mask = (pupil_df['times'] >= win_start) & (pupil_df['times'] <= win_end)
        t_raw = pupil_df.loc[mask, 'times'].values
        p_zscored = pupil_df.loc[mask, 'session_zscore'].values
        
        if len(t_raw) < 10 or np.isnan(p_zscored).all():
            all_pupil_sizes.append(None)
            all_pupil_times.append(None)
            continue

        all_pupil_sizes.append(p_zscored)
        all_pupil_times.append(t_raw)

    masked_trials_df['eid'] = eid
    masked_trials_df['pupil_size'] = all_pupil_sizes
    masked_trials_df['pupil_timestamps'] = all_pupil_times

    return masked_trials_df


def create_pupilometry_data(one, eid, engagement_df, scalar_motivation, action_kernel):
    sess = SessionLoader(one, eid)
    sess.load_pupil()

    trials_df, mask = load_trials_and_mask(one, eid)
    animal_engagement = engagement_df[engagement_df['eid']==eid].reset_index(drop=True)
    stimulus_engage = scalar_motivation[eid]
    akernel_df = action_kernel[eid]


    trials_df = trials_df.merge(animal_engagement[['p_state1','p_state2','signed_contrast','rewarded']],left_index=True, right_index=True)
    trials_df['akernel_prior'] = akernel_df['prior']
    trials_df['pe_right'] = akernel_df['prediction_error_right']
    trials_df['motivation_scalar'] = stimulus_engage

    masked_trials_df = trials_df[mask].copy()
    pupil_df = sess.pupil
    
    assert np.all(masked_trials_df['feedbackType']==masked_trials_df['rewarded'])

    window = (-0.250,0)

    masked_trials_df = masked_trials_df.reset_index(drop=True) #keep everything
    mean_pupil_sizes = np.full(len(masked_trials_df), np.nan)
    deviation_pupil_sizes = np.full(len(masked_trials_df), np.nan)

    for index, trial in masked_trials_df.iterrows():
        window_start = trial['stimOn_times'] + window[0] 
        window_end = trial['stimOn_times'] + window[1]
            
        time_mask = (pupil_df['times'] >= window_start) & (pupil_df['times'] <= window_end)
        
        window_data = pupil_df.loc[time_mask, 'pupilDiameter_raw']
        if not window_data.empty: # type: ignore
            mean_pupil_sizes[index] = np.nanmean(window_data) # type: ignore
            deviation_pupil_sizes[index] = np.nanstd(window_data) # type: ignore
    masked_trials_df['pupil_mean'] = mean_pupil_sizes
    masked_trials_df['pupil_std']  = deviation_pupil_sizes
    masked_trials_df['signed_contrast'] = np.nan_to_num(masked_trials_df['contrastLeft']) - np.nan_to_num(masked_trials_df['contrastRight'])
    
    masked_trials_df['response'] = masked_trials_df['response_times'] - masked_trials_df['stimOn_times'] 
    masked_trials_df['eid'] = eid

    return trials_df, masked_trials_df


def run_single_session(one , eid):
    

    sess = SessionLoader(one, eid)
    sess.load_pupil()

    trials, mask = load_trials_and_mask(one, eid)

    trials_df = trials[mask].copy()
    pupil_df = sess.pupil
    is_contrast_significant, _ = check_session_significance(trials_df, pupil_df, (1,2), plot=False)
    is_feedback_significant = test_feedback_modulation(trials_df, pupil_df, False)

    return is_contrast_significant, is_feedback_significant


if __name__ == '__main__':

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    bwm_df = bwm_query()
    eids = bwm_df['eid'].unique()

    # pupil_qc = {}

    # for session_id in eids:
    #     try:
    #         cs, fs = run_single_session(one, session_id)
    #         pupil_qc[session_id] = [cs,fs]
    #     except Exception as e:
    #         print(e)
    #         pupil_qc[session_id] = [np.nan, np.nan]

    # print(cs,fs)    
    # df = pd.DataFrame.from_dict(pupil_qc, orient='index', columns=['contrast','feedback'])

    # df.to_parquet('./data/generated/pupil_qc.pqt')

    pupil_qc = pd.read_parquet('./data/generated/pupil_qc.pqt')
    engagement_df = pd.read_parquet('./data/external/merged_behavioral_and_states.pqt') # this is the glm engagement

    with open('./data/external/all_eids_engagement.pkl','rb') as f:
        scalar_motivation = pkl.load(f)
    
    with open('./data/external/action_kernel.pkl','rb') as f:
        akernel = pkl.load(f)

    complete_df = []
    # complete_pickle = {}
    for session_id in eids:
        try:
            contrast_modulation = pupil_qc.loc[session_id]['contrast']
            if contrast_modulation:
                #subset_df_complete, subset_df_masked = create_pupilometry_data(one, session_id, engagement_df, scalar_motivation, akernel)
                subset_df_masked = feedback_pupilometry(one, session_id, engagement_df, scalar_motivation, akernel)
                complete_df.append(subset_df_masked)
                # complete_pickle[session_id] = subset_df_complete 
        except Exception as e:
            print(e)

    try:
        complete_df = pd.concat(complete_df)
        complete_df.to_parquet('./data/complete_pupil_df_feedback.pqt')

        # with open('./data/generated/complete_df_untruncated.pkl','wb') as f:
        #     pkl.dump(complete_pickle, f)

    except Exception as e:
        with open('./data/generated/complete_pupil_df_pickl_feedback.pkl','wb') as f:
            pkl.dump(complete_df, f)
        print('pickle dump for some reason')
