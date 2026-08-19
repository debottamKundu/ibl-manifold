import os
import glob
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pathlib

def fisher_z_transform(r):
    # Clip r to avoid inf in arctanh
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)

def inverse_fisher_z(z):
    return np.tanh(z)

def load_and_pair_logits(data_dir, stim_region, choice_region, acc_threshold=0.0):
    """
    Loads spatial alignment logits from all sessions in the directory and pairs them up.
    Returns dictionaries containing data for correct and incorrect trials.
    Filters out sessions where the balanced accuracy for either region is below acc_threshold.
    """
    pkl_files = glob.glob(os.path.join(data_dir, "*_spatial_alignment_logits.pkl"))
    
    paired_data_correct = []
    paired_data_incorrect = []
    
    for pkl_file in pkl_files:
        session_id = os.path.basename(pkl_file).split("_")[0]
        
        with open(pkl_file, "rb") as f:
            results = pkl.load(f)
            
        stim_res = next((r for r in results if r["region"] == stim_region and r["epoch"] == "stim"), None)
        choice_res = next((r for r in results if r["region"] == choice_region and r["epoch"] == "choice"), None)
        
        if stim_res is None or choice_res is None:
            continue
            
        if acc_threshold > 0.0:
            if stim_res.get("cv_balanced_accuracy", 0) < acc_threshold or choice_res.get("cv_balanced_accuracy", 0) < acc_threshold:
                continue
                
        stim_c_idx = stim_res["correct_trial_indices"]
        choice_c_idx = choice_res["correct_trial_indices"]
        
        c_intersect, stim_c_ind, choice_c_ind = np.intersect1d(stim_c_idx, choice_c_idx, return_indices=True)
        
        if len(c_intersect) > 0:
            stim_logits_c = stim_res["correct_logits"][stim_c_ind]
            choice_logits_c = choice_res["correct_logits"][choice_c_ind]
            true_labels_c = stim_res["correct_true_labels"][stim_c_ind] # Same for both
            
            paired_data_correct.append({
                "session": session_id,
                "stim_logits": stim_logits_c,
                "choice_logits": choice_logits_c,
                "true_labels": true_labels_c
            })
            
        if stim_res["incorrect_trial_indices"] is not None and choice_res["incorrect_trial_indices"] is not None:
            stim_i_idx = stim_res["incorrect_trial_indices"]
            choice_i_idx = choice_res["incorrect_trial_indices"]
            
            i_intersect, stim_i_ind, choice_i_ind = np.intersect1d(stim_i_idx, choice_i_idx, return_indices=True)
            
            if len(i_intersect) > 0:
                stim_logits_i = stim_res["incorrect_logits"][stim_i_ind]
                choice_logits_i = choice_res["incorrect_logits"][choice_i_ind]
                true_labels_i = stim_res["incorrect_true_labels"][stim_i_ind]
                
                paired_data_incorrect.append({
                    "session": session_id,
                    "stim_logits": stim_logits_i,
                    "choice_logits": choice_logits_i,
                    "true_labels": true_labels_i
                })
                
    return paired_data_correct, paired_data_incorrect

def plot_alignment(data_dir, stim_region, choice_region, output_dir="./figures/spatial_alignment", acc_threshold=0.0):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    paired_c, paired_i = load_and_pair_logits(data_dir, stim_region, choice_region, acc_threshold)
    
    if not paired_c:
        print(f"No paired data found for {stim_region} (stim) and {choice_region} (choice).")
        return
        
    print(f"Found {len(paired_c)} sessions with paired correct trials.")
    

    corrs_c = []
    corrs_i = []
    session_ids = []
    
    for sess_data_c in paired_c:
        sess_id = sess_data_c["session"]
        

        sess_data_i = next((d for d in paired_i if d["session"] == sess_id), None)
        
        if sess_data_i is not None and len(sess_data_i["stim_logits"]) > 2 and len(sess_data_c["stim_logits"]) > 2:
            r_c, _ = pearsonr(sess_data_c["stim_logits"], sess_data_c["choice_logits"])
            r_i, _ = pearsonr(sess_data_i["stim_logits"], sess_data_i["choice_logits"])
            
            if not np.isnan(r_c) and not np.isnan(r_i):
                corrs_c.append(r_c)
                corrs_i.append(r_i)
                session_ids.append(sess_id)
            
    mean_corr_c = inverse_fisher_z(np.nanmean(fisher_z_transform(corrs_c))) if corrs_c else np.nan
    mean_corr_i = inverse_fisher_z(np.nanmean(fisher_z_transform(corrs_i))) if corrs_i else np.nan
    
    print(f"Matched {len(session_ids)} sessions for paired statistics.")
    print(f"Session-averaged Pearson R (Correct): {mean_corr_c:.3f}")
    print(f"Session-averaged Pearson R (Incorrect): {mean_corr_i:.3f}")
    
    t_stat = np.nan
    p_val = np.nan
    if len(session_ids) > 1:
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(fisher_z_transform(corrs_c), fisher_z_transform(corrs_i))
        print(f"Delta Metric Significance (Paired t-test): t={t_stat:.3f}, p={p_val:.3e}")
    else:
        print("Not enough matched sessions for a paired t-test.")
        
    # 1b. Check decoder performance on incorrect trials (Stim vs Choice)
    acc_i_stim = []
    acc_i_choice = []
    from sklearn.metrics import balanced_accuracy_score
    
    for sess_data in paired_i:
        stim_preds = np.where(sess_data["stim_logits"] > 0, 1, -1)
        choice_preds = np.where(sess_data["choice_logits"] > 0, 1, -1)
        true_labels = sess_data["true_labels"]
        
        try:
            a_s = balanced_accuracy_score(true_labels, stim_preds)
            a_c = balanced_accuracy_score(true_labels, choice_preds)
            acc_i_stim.append(a_s)
            acc_i_choice.append(a_c)
        except ValueError:
            pass

    mean_acc_stim = np.nan
    mean_acc_choice = np.nan
    t_stat_acc = np.nan
    p_val_acc = np.nan
    
    if len(acc_i_stim) > 1:
        from scipy.stats import ttest_rel
        mean_acc_stim = np.mean(acc_i_stim)
        mean_acc_choice = np.mean(acc_i_choice)
        t_stat_acc, p_val_acc = ttest_rel(acc_i_stim, acc_i_choice)
        print(f"\n--- Incorrect Trials Decoding Performance ---")
        print(f"Mean Stim Region Accuracy: {mean_acc_stim:.3f}")
        print(f"Mean Choice Region Accuracy: {mean_acc_choice:.3f}")
        print(f"Stim vs Choice Difference (Paired t-test): t={t_stat_acc:.3f}, p={p_val_acc:.3e}\n")

    
    # 2. Pooled Normalized Logits for Visualization
    all_stim_c, all_choice_c, all_labels_c = [], [], []
    all_stim_i, all_choice_i, all_labels_i = [], [], []
    
    for sess_data in paired_c:
        s_norm = (sess_data["stim_logits"] - np.mean(sess_data["stim_logits"])) / (np.std(sess_data["stim_logits"]) + 1e-8)
        c_norm = (sess_data["choice_logits"] - np.mean(sess_data["choice_logits"])) / (np.std(sess_data["choice_logits"]) + 1e-8)
        
        all_stim_c.extend(s_norm)
        all_choice_c.extend(c_norm)
        all_labels_c.extend(sess_data["true_labels"])
        
    for sess_data in paired_i:
        s_norm = (sess_data["stim_logits"] - np.mean(sess_data["stim_logits"])) / (np.std(sess_data["stim_logits"]) + 1e-8)
        c_norm = (sess_data["choice_logits"] - np.mean(sess_data["choice_logits"])) / (np.std(sess_data["choice_logits"]) + 1e-8)
        
        all_stim_i.extend(s_norm)
        all_choice_i.extend(c_norm)
        all_labels_i.extend(sess_data["true_labels"])
        
    df_c = pd.DataFrame({"Stim Logits": all_stim_c, "Choice Logits": all_choice_c, "Stim Side": all_labels_c, "Outcome": "Correct"})
    df_i = pd.DataFrame({"Stim Logits": all_stim_i, "Choice Logits": all_choice_i, "Stim Side": all_labels_i, "Outcome": "Incorrect"})
    
    df = pd.concat([df_c, df_i], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    
    for i, (outcome, ax) in enumerate(zip(["Correct", "Incorrect"], axes)):
        subset = df[df["Outcome"] == outcome]
        
        if len(subset) == 0:
            ax.set_title(f"{outcome} Trials (No Data)")
            continue
            
        n_trials = len(subset)
        r_val = mean_corr_c if outcome == "Correct" else mean_corr_i
        
        # if n_trials < 1000:
        sns.scatterplot(data=subset, x="Stim Logits", y="Choice Logits", hue="Stim Side", 
                            palette="coolwarm", alpha=0.7, ax=ax, s=20)
        sns.regplot(data=subset, x="Stim Logits", y="Choice Logits", scatter=False, ax=ax, color='black', line_kws={'linestyle': '--', 'linewidth': 1.5})
        # else:
            # For dense data, use KDE plot and lay a light scatter under it
            # sns.scatterplot(data=subset, x="Stim Logits", y="Choice Logits", hue="Stim Side", 
            #                 palette="coolwarm", alpha=0.2, ax=ax, s=10, legend=False)
            # sns.kdeplot(data=subset, x="Stim Logits", y="Choice Logits", 
            #             cmap="gray", alpha=0.6, ax=ax, linewidths=1)
                        
        
            
        ax.set_title(f"{outcome} Trials , r = {r_val:.3f}")
        # ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        # ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        
    plt.suptitle(f"{stim_region} to {choice_region}")
    sns.despine()
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f"{stim_region}_{choice_region}_alignment.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_file}")
    plt.close()
    
    return {
        "mean_corr_correct": mean_corr_c,
        "mean_corr_incorrect": mean_corr_i,
        "t_stat_corr_delta": t_stat,
        "p_val_corr_delta": p_val,
        "mean_acc_stim_incorrect": mean_acc_stim,
        "mean_acc_choice_incorrect": mean_acc_choice,
        "t_stat_acc_delta": t_stat_acc,
        "p_val_acc_delta": p_val_acc,
        "n_sessions": len(session_ids)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Spatial Alignment")
    parser.add_argument("--data_dir", type=str, default="./data/generated/wifi/spatial_alignment")
    parser.add_argument("--stim", type=str, default="VISp", help="Stimulus epoch region")
    parser.add_argument("--choice", type=str, default="MOp", help="Choice epoch region")
    parser.add_argument("--acc_threshold", type=float, default=0.0, help="Minimum balanced accuracy required for both regions to include the session (e.g., 0.55)")
    parser.add_argument("--batch_csv", type=str, default=None, help="Path to significant_pairs.csv to run in batch mode")
    
    args = parser.parse_args()
    
    if args.batch_csv:
        df_pairs = pd.read_csv(args.batch_csv)
        if 'is_significant' in df_pairs.columns:
            df_pairs = df_pairs[df_pairs['is_significant'] == True]
            
        batch_stats = []
        for _, row in df_pairs.iterrows():
            stim = row['seed']
            choice = row['target']
            print(f"\n================ Processing pair: {stim} -> {choice} ================")
            stats = plot_alignment(args.data_dir, stim, choice, acc_threshold=args.acc_threshold)
            if stats:
                stats.update({'stim': stim, 'choice': choice})
                batch_stats.append(stats)
                
        if batch_stats:
            stats_df = pd.DataFrame(batch_stats)
            out_csv = os.path.join(args.data_dir, "batch_spatial_alignment_stats.csv")
            stats_df.to_csv(out_csv, index=False)
            print(f"\nBatch run complete. Summary stats saved to {out_csv}")
    else:
        plot_alignment(args.data_dir, args.stim, args.choice, acc_threshold=args.acc_threshold)
