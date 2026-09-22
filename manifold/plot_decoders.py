import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(pattern):
    """Loads all parquet files matching the pattern and appends a session_id based on filename."""
    files = glob.glob(pattern)
    df_list = []
    for f in files:
        uuid = os.path.basename(f).split('_')[0]
        d = pd.read_parquet(f)
        d['session_id'] = uuid
        df_list.append(d)
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def aggregate_crosstime(df):
    """Aggregates crosstime decoders data by melting time frames."""
    time_cols = [c for c in df.columns if c.startswith('ridge_projection_f')]
    id_vars = ['session_id', 'model_epoch', 'stim_side_computed', 'choice', 'is_correct_trial', 'is_congruent']
    df_sub = df[id_vars + time_cols].copy()
    
    def get_cond(row):
        if row['is_correct_trial'] and row['is_congruent']: return 'Correct Congruent'
        if row['is_correct_trial'] and not row['is_congruent']: return 'Correct Incongruent'
        if not row['is_correct_trial'] and row['is_congruent']: return 'Incorrect Congruent'
        if not row['is_correct_trial'] and not row['is_congruent']: return 'Incorrect Incongruent'
        return 'Unknown'
        
    df_sub['condition'] = df_sub.apply(get_cond, axis=1)
    
    df_all_correct = df_sub[df_sub['is_correct_trial'] == True].copy()
    df_all_correct['condition'] = 'All Correct'
    
    df_combined = pd.concat([df_sub, df_all_correct], ignore_index=True)
    
    df_melt = df_combined.melt(id_vars=['session_id', 'model_epoch', 'stim_side_computed', 'choice', 'condition', 'is_correct_trial'], 
                               value_vars=time_cols, 
                               var_name='frame', value_name='projection')
    df_melt['time'] = df_melt['frame'].str.extract(r'f(-?\d+)').astype(int)
    
    return df_melt

def plot_crosstime(df_melt, show_ci=True):
    """
    Plots the crosstime decoders.
    Choice and stim are in different subplots/figures.
    Conditions are columns.
    Left/Right are rows.
    """
    errorbar_kw = ('ci', 95) if show_ci else None
    
    for epoch in ['stim', 'choice']:
        df_epoch = df_melt[df_melt['model_epoch'] == epoch].copy()
        if df_epoch.empty:
            continue
            
        if epoch == 'stim':
            df_epoch['side'] = df_epoch['stim_side_computed'].map({-1.0: 'Left Stimulus', 1.0: 'Right Stimulus'})
        else:
            df_epoch['side'] = df_epoch['choice'].map({-1.0: 'Left Choice', 1.0: 'Right Choice'})
            
        df_epoch = df_epoch[df_epoch['side'].notna()]
            
        def get_color(cond):
            if 'Incorrect' in cond: return 'red'
            return 'green'
            
        palette = {cond: get_color(cond) for cond in df_epoch['condition'].unique()}
        
        df_session_avg = df_epoch.groupby(['session_id', 'condition', 'side', 'time'])['projection'].mean().reset_index()
        
        g = sns.FacetGrid(df_session_avg, col="condition", row="side", margin_titles=True, 
                          col_order=['Correct Congruent', 'Correct Incongruent', 'Incorrect Congruent', 'Incorrect Incongruent', 'All Correct'],
                          sharey=True, sharex=True, height=3, aspect=1.2)
        
        g.map_dataframe(sns.lineplot, x='time', y='projection', hue='condition', palette=palette, errorbar=errorbar_kw, estimator='mean', marker='o')
        
        # Vertical line for training frame
        vline_x = 1 if epoch == 'stim' else 4
        g.map(plt.axvline, x=vline_x, color='k', linestyle='--', alpha=0.5)
        
        g.set_axis_labels("Frames", "Projection")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        sns.despine(fig=g.fig)
        g.fig.suptitle(f"{epoch} decoders", y=1.02)
        
        plt.tight_layout()
        # g.fig.savefig(f"crosstime_{epoch}.png", bbox_inches='tight')
        # plt.close(g.fig)

def aggregate_prior(df, region='VISp'):
    """Aggregates prior decoders data."""
    time_cols = [c for c in df.columns if c.startswith('logreg_prob_f')]
    id_vars = ['session_id', 'region', 'is_correct_trial', 'is_congruent', 'prior']
    df_sub = df[id_vars + time_cols].copy()
    
    # Filter out unbiased blocks
    df_sub = df_sub[df_sub['prior'] != 0.5].copy()
    
    df_sub = df_sub[~df_sub['is_congruent']].copy()
    
    # Filter for region
    df_sub = df_sub[df_sub['region'] == region].copy()
    
    def get_cond(row):
        if row['is_correct_trial']: return 'Correct'
        return 'Error'
        
    df_sub['condition'] = df_sub.apply(get_cond, axis=1)
    
    df_melt = df_sub.melt(id_vars=['session_id', 'region', 'condition', 'prior'], 
                          value_vars=time_cols, 
                          var_name='frame', value_name='prob')
    df_melt['time'] = df_melt['frame'].str.extract(r'f(-?\d+)').astype(int)
    
    df_melt['true_block'] = np.where(df_melt['prior'] > 0.5, 1, -1)
    df_melt['aligned_prob'] = np.where(df_melt['true_block'] == 1, df_melt['prob'], 1 - df_melt['prob'])
    
    return df_melt

def plot_prior(df_melt, region='VISp', show_ci=False):
    errorbar_kw = 'se' if show_ci else None
    
    df_session_avg = df_melt.groupby(['session_id', 'region', 'condition', 'time'])['aligned_prob'].mean().reset_index()
    
    palette = {'Correct': 'green', 'Error': 'red'}
    
    g = sns.FacetGrid(df_session_avg, col="region", margin_titles=True, height=4, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='time', y='aligned_prob', hue='condition', palette=palette, errorbar=errorbar_kw, err_style='bars', estimator='mean', marker='o')
    g.map(plt.axhline, y=0.5, color='gray', linestyle=':')
    g.map(plt.axvline, x=-2, color='black', linestyle='--', alpha=0.5, label='Training Frame')
    g.set(ylim=(0.4, 0.9))
    g.add_legend()
    g.set_axis_labels("Frames", "True Block Probability")
    sns.despine(fig=g.fig)
    g.fig.suptitle(f"{region} Incongruent Trials", y=1.05)
    
    plt.tight_layout()
    # g.fig.savefig(f"prior_decoders_{region.lower()}.png", bbox_inches='tight')
    # plt.close(g.fig)

def plot_prior_per_animal(df_melt, region='VISp', show_ci=False):
    errorbar_kw = 'se' if show_ci else None
    
    palette = {'Correct': 'green', 'Error': 'red'}
    
    g = sns.FacetGrid(df_melt, col="session_id", col_wrap=4, margin_titles=True, height=3, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='time', y='aligned_prob', hue='condition', palette=palette, errorbar=errorbar_kw, err_style='bars', estimator='mean', marker='o')
    g.map(plt.axhline, y=0.5, color='gray', linestyle=':')
    g.map(plt.axvline, x=-2, color='black', linestyle='--', alpha=0.5, label='Training Frame')
    g.set(ylim=(0.4, 0.9))
    g.add_legend()
    g.set_axis_labels("Time (frames)", "Probability of True Block")
    g.set_titles(col_template="{col_name}")
    sns.despine(fig=g.fig)
    g.fig.suptitle(f"{region} - Logistic Regression (Per Animal)", y=1.05)
    
    plt.tight_layout()
    # g.fig.savefig(f"prior_decoders_{region.lower()}_per_animal.png", bbox_inches='tight')
    # plt.close(g.fig)

if __name__ == '__main__':
    print("Loading crosstime decoders...")
    df_cross = load_data('../data/generated/crosstime_decoders/allcorrect_trained/*.pqt')
    if not df_cross.empty:
        df_cross_melt = aggregate_crosstime(df_cross)
        print("Plotting crosstime decoders...")
        plot_crosstime(df_cross_melt)
    else:
        print("No crosstime data found.")
        
    print("Loading temporal prior...")
    df_prior = load_data('../data/generated/temporal_prior/*.pqt')
    if not df_prior.empty:
        for r in ['VISp', 'MOs']:
            df_prior_melt_r = aggregate_prior(df_prior, region=r)
            if not df_prior_melt_r.empty:
                print(f"Plotting temporal prior {r}...")
                plot_prior(df_prior_melt_r, region=r, show_ci=True)
                plot_prior_per_animal(df_prior_melt_r, region=r, show_ci=True)
    else:
        print("No temporal prior data found.")
        
    print("Done!")
