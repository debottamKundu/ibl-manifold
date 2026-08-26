import argparse
import os
import sys
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import traceback

warnings.filterwarnings("ignore")

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import load_trials_and_mask

from communication_subspace.ibl_communication.utils import load_widefield_epoch, setup_logger
from communication_subspace.ibl_communication.crossvalidated_rrr import optimize_rrr_rank
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle as pkl
from glob import glob
import seaborn as sns
from manifold.widefield_ppi import return_labels
from matplotlib.lines import Line2D

def combine_data_frames(file_list):
    temp_df = []
    for idx,file in enumerate(file_list):
        df = pd.read_parquet(file)
        df['subject-id'] = idx
        temp_df.append(df)
    temp_df = pd.concat(temp_df)
    return temp_df


def prepare_and_aggregate(df_raw, projection_prefix='ridge_projection_'):
    """
    Transforms raw IBL trial data into aggregated stimulus and choice dataframes.
    Routes rows based on 'model_epoch' ('stim' vs 'choice') to ensure accurate 
    temporal alignment before subject-level averaging.
    """
    df = df_raw.copy()

    subject_col = 'subject-id'

    df['Outcome'] = np.where(df['is_correct_trial'] == 1, 'Correct', 'Error')
    df['Congruency'] = np.where(df['is_congruent'] == 1, 'Congruent', 'Incongruent')
    
    proj_cols = [f'{projection_prefix}f{i}' for i in range(5)]
    id_variables = [subject_col, 'Outcome', 'Congruency', 'Side']
    
    df['model_epoch'] = df['model_epoch'].astype(str).str.lower()
    df_stim_raw = df[(df['model_epoch'] == 'stim') & (df['signed_contrast'] != 0)].copy()
    df_stim_raw['Side'] = np.where(df_stim_raw['signed_contrast'] < 0, 'Left', 'Right')
    
    df_stim_melt = pd.melt(
        df_stim_raw,
        id_vars=id_variables,
        value_vars=proj_cols,
        var_name='frame',
        value_name='distance'
    )
    
    stim_time_map = {f'{projection_prefix}f{i}': i for i in range(5)}
    df_stim_melt['time'] = df_stim_melt['frame'].map(stim_time_map)
    
    df_stim_agg = df_stim_melt.groupby(id_variables + ['time'])['distance'].mean().reset_index()

    df_choice_raw = df[df['model_epoch'] == 'choice'].copy()
    df_choice_raw['Side'] = df_choice_raw['choice'].map({-1: 'Left', 1: 'Right'})
    df_choice_raw = df_choice_raw.dropna(subset=['Side'])
    
    df_choice_melt = pd.melt(
        df_choice_raw,
        id_vars=id_variables,
        value_vars=proj_cols,
        var_name='frame',
        value_name='distance'
    )
    
    choice_time_map = {f'{projection_prefix}f{i}': i - 4 for i in range(5)}
    df_choice_melt['time'] = df_choice_melt['frame'].map(choice_time_map)
    
    df_choice_agg = df_choice_melt.groupby(id_variables + ['time'])['distance'].mean().reset_index()

    return df_stim_agg, df_choice_agg

def prepare_extremes_only(df_raw, projection_prefix='ridge_projection_'):
    df = df_raw.copy()
    
    subject_col = 'subject-id'

    df['Outcome'] = np.where(df['is_correct_trial'] == 1, 'Correct', 'Error')
    df['Congruency'] = np.where(df['is_congruent'] == 1, 'Congruent', 'Incongruent')
    df['model_epoch'] = df['model_epoch'].astype(str).str.lower()
    
    df['contrast_abs'] = df['signed_contrast'].abs()
    valid_contrasts = df[df['contrast_abs'] > 0]['contrast_abs'].unique()
    min_c, max_c = valid_contrasts.min(), valid_contrasts.max()
    
    df = df[df['contrast_abs'].isin([min_c, max_c])].copy()
    
    df['Contrast_Level'] = np.where(df['contrast_abs'] == max_c, 'High', 'Low')
    df['Outcome_Group'] = df['Outcome'] + ' - ' + df['Contrast_Level']
    
    proj_cols = [f'{projection_prefix}f{i}' for i in range(5)]
    id_variables = [subject_col, 'Outcome_Group', 'Contrast_Level', 'Congruency', 'Side']

    df_stim_raw = df[(df['model_epoch'] == 'stim')].copy()
    df_stim_raw['Side'] = np.where(df_stim_raw['signed_contrast'] < 0, 'Left', 'Right')
    
    df_stim_melt = pd.melt(
        df_stim_raw, id_vars=id_variables, value_vars=proj_cols,
        var_name='frame', value_name='distance'
    )
    df_stim_melt['time'] = df_stim_melt['frame'].map({f'{projection_prefix}f{i}': i for i in range(5)})
    df_stim_agg = df_stim_melt.groupby(id_variables + ['time'])['distance'].mean().reset_index()

    df_choice_raw = df[(df['model_epoch'] == 'choice')].copy()
    df_choice_raw['Side'] = df_choice_raw['choice'].map({-1: 'Left', 1: 'Right'})
    df_choice_raw = df_choice_raw.dropna(subset=['Side'])
    
    df_choice_melt = pd.melt(
        df_choice_raw, id_vars=id_variables, value_vars=proj_cols,
        var_name='frame', value_name='distance'
    )
    df_choice_melt['time'] = df_choice_melt['frame'].map({f'{projection_prefix}f{i}': i - 4 for i in range(5)})
    df_choice_agg = df_choice_melt.groupby(id_variables + ['time'])['distance'].mean().reset_index()

    return df_stim_agg, df_choice_agg



def plot_decoder_extremes(df_stim, df_choice, title="Decoder Projections"):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), sharey='row')
    
    
    palette_map = {
        'Correct - High': '#006400',  # Dark Green
        'Correct - Low': '#32CD32',   # Lime Green
        'Error - High': '#8B0000',    # Dark Red
        'Error - Low': '#FF6347'      # Tomato/Light Red
    }
    
    size_map = {'High': 3.5, 'Low': 1.5}
    style_map = {'Left': '', 'Right': (3, 3)} 
    
    conditions = [
        ('Correct', lambda df: df[df['Outcome_Group'].str.contains('Correct')]),
        ('Congruent Error', lambda df: df[(df['Outcome_Group'].str.contains('Error')) & (df['Congruency'] == 'Congruent')]),
        ('Incongruent Error', lambda df: df[(df['Outcome_Group'].str.contains('Error')) & (df['Congruency'] == 'Incongruent')])
    ]
    
    row_data = [
        (0, df_stim, 'Stimulus Decoder : VISp', 'Frames'),
        (1, df_choice, 'Choice Decoder : MOs', 'Frames')
    ]
    
    for row_idx, df_row, ylabel, xlabel in row_data:
        for col_idx, (cond_name, filter_func) in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            subset = filter_func(df_row)
            
            if subset.empty:
                ax.axis('off')
                continue
                
            sns.lineplot(
                data=subset, x='time', y='distance',
                hue='Outcome_Group', style='Side', size='Contrast_Level',
                palette=palette_map, sizes=size_map, dashes=style_map,
                markers=True, markersize=7, errorbar=None, ax=ax, legend=False
            )
            
            if row_idx==0:
                ax.axvline(1, color='gray', linestyle=':', linewidth=1)
            else:
                ax.axvline(0, color='gray', linestyle=':', linewidth=1)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            time_points = sorted(subset['time'].unique())
            ax.set_xticks(time_points)
            
            if row_idx == 0: ax.set_title(cond_name, fontsize=14, pad=15)
            if col_idx == 0: ax.set_ylabel(ylabel, fontsize=12)
            else: ax.set_ylabel('') 
            if row_idx == 1: ax.set_xlabel(xlabel, fontsize=12)
            else: ax.set_xlabel('')
                
    custom_lines = [
        Line2D([0], [0], color='#006400', lw=3.5, label='Contrast = 1'),
        Line2D([0], [0], color='#32CD32', lw=1.5, label='Contrast = 0.0625'),
        Line2D([0], [0], color='#8B0000', lw=3.5, label='Contrast = 1'),
        Line2D([0], [0], color='#FF6347', lw=1.5, label='Contrast = 0.0625'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o', label='Left Side'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', marker='X', label='Right Side')
    ]
    fig.legend(handles=custom_lines, loc='center right', bbox_to_anchor=(1.15, 0.5), title='Legend')
    
    fig.suptitle(title, fontsize=16, y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_decoder_evolution(df_stim, df_choice, title="Decoder Projections"):
    """
    Plots a 2x3 grid: 
    Rows: Stimulus (Top), Choice (Bottom)
    Cols: Correct, Congruent Error, Incongruent Error
    Displays explicit discrete markers for the 5 frames.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharey='row')
    
    palette_map = {'Correct': 'green', 'Error': 'red'}
    style_map = {'Left': '', 'Right': (3, 3)} 
    
    conditions = [
        ('Correct', lambda df: df[df['Outcome'] == 'Correct']),
        ('Congruent Error', lambda df: df[(df['Outcome'] == 'Error') & (df['Congruency'] == 'Congruent')]),
        ('Incongruent Error', lambda df: df[(df['Outcome'] == 'Error') & (df['Congruency'] == 'Incongruent')])
    ]
    
    row_data = [
        (0, df_stim, 'Stimulus Decoder : VISp)', 'Frames'),
        (1, df_choice, 'Choice Decoder : MOs)', 'Frames')
    ]
    
    for row_idx, df_row, ylabel, xlabel in row_data:
        for col_idx, (cond_name, filter_func) in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            subset = filter_func(df_row)
            
            if subset.empty:
                ax.axis('off')
                continue
                
            sns.lineplot(
                data=subset,
                x='time',
                y='distance',
                hue='Outcome',
                style='Side',
                palette=palette_map,
                dashes=style_map,
                markers=True,         # Draws explicit data points
                markersize=8,         # Makes the points easily visible
                errorbar=('ci', 95),
                linewidth=2,
                ax=ax,
                legend=False
            )
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            if row_idx==0:
                ax.axvline(1, color='gray', linestyle=':', linewidth=1)
            else:
                ax.axvline(0, color='gray', linestyle=':', linewidth=1)
            
            # Force x-axis to only show the exact discrete frames we have
            time_points = sorted(subset['time'].unique())
            ax.set_xticks(time_points)
            
            if row_idx == 0:
                ax.set_title(cond_name, fontsize=14, pad=15)
                
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            else:
                ax.set_ylabel('') 
                
            if row_idx == 1:
                ax.set_xlabel(xlabel, fontsize=12)
            else:
                ax.set_xlabel('')
                
    # Custom legend incorporating markers
    custom_lines = [
        Line2D([0], [0], color='green', lw=2, marker='o', label='Correct'),
        Line2D([0], [0], color='red', lw=2, marker='X', label='Error'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o', label='Left'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', marker='X', label='Right')
    ]
    fig.legend(handles=custom_lines, loc='center right', bbox_to_anchor=(1.1, 0.5), title='Legend')
    
    fig.suptitle(title, fontsize=16, y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.show()