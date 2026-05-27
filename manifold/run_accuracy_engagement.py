import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import ot
import argparse
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance


def find_top_sessions(engagement_signal_dict, target_eid, top_n=200):
    distances = []
    target_distribution = engagement_signal_dict[target_eid]
    for eid, engagement in engagement_signal_dict.items():
        if eid == target_eid:
            continue
        source_distribution = engagement_signal_dict[eid]
        w_dist = wasserstein_distance(source_distribution, target_distribution)
        distances.append({"session_id": eid, "distance": w_dist})
    df = pd.DataFrame(distances)
    df = df.sort_values(by="distance")
    return df.reset_index().head(top_n)


def generate_null_interpolation(target_signal, source_signal):
    target = np.asarray(target_signal).flatten()
    source = np.asarray(source_signal).flatten()

    N = len(target)
    M = len(source)

    time_target = np.linspace(0, 1, N)
    time_source = np.linspace(0, 1, M)

    source_interpolated = np.interp(time_target, time_source, source)

    valid_target_mask = ~np.isnan(target)
    valid_source_mask = ~np.isnan(source_interpolated)

    clean_target = target[valid_target_mask].reshape(-1, 1)
    clean_source = source_interpolated[valid_source_mask].reshape(-1, 1)

    ot_mapper = ot.da.EMDTransport()
    ot_mapper.fit(Xs=clean_source, Xt=clean_target)
    transformed_clean_source = ot_mapper.transform(Xs=clean_source).flatten()

    final_null_signal = np.full(N, np.nan)
    final_null_signal[valid_source_mask] = transformed_clean_source

    return final_null_signal


def get_accuracy_engagement(
    target, pred, trials_df, engagement_values=None, engagement_col="engagement"
):
    if engagement_values is not None:
        eng_vals = np.asarray(engagement_values).flatten()
        eng_vals = eng_vals[trials_df["mask"]]
    elif trials_df is not None and engagement_col in trials_df.columns:
        eng_vals = trials_df[engagement_col][trials_df["mask"]].values
    else:
        raise ValueError("Must provide 'engagement_values' or 'trials_df'")

    median_val = np.nanmedian(eng_vals)
    engagement_bin = eng_vals > median_val

    # Align targets and predictions
    offset = 1 - target.max()
    corr_test = target + offset
    corr_pred = pred + offset

    # Exclude 0 contrast
    valid_mask = corr_test != 0
    corr_test = corr_test[valid_mask]
    corr_pred = corr_pred[valid_mask]
    engagement_bin = engagement_bin[valid_mask]

    pred_signs = np.sign(corr_pred)
    target_signs = np.sign(corr_test)

    correct = pred_signs == target_signs

    df = pd.DataFrame(
        {"contrast": corr_test, "correct": correct, "engagement_bin": engagement_bin}
    )

    # Group by both contrast and engagement_bin
    grpby = df.groupby(["contrast", "engagement_bin"])["correct"].mean().unstack("engagement_bin")

    contrasts_unique = sorted(np.unique(corr_test))

    acc_low = {}
    acc_high = {}

    for c in contrasts_unique:
        if c in grpby.index:
            if False in grpby.columns:
                acc_low[c] = grpby.loc[c, False]
            else:
                acc_low[c] = np.nan
            if True in grpby.columns:
                acc_high[c] = grpby.loc[c, True]
            else:
                acc_high[c] = np.nan
        else:
            acc_low[c] = np.nan
            acc_high[c] = np.nan

    return acc_low, acc_high


def plot_grouped_accuracy_bar(med_real_low_acc, med_real_high_acc, save_path):
    contrasts = sorted(list(med_real_low_acc.keys()))
    low_means = [med_real_low_acc[c] for c in contrasts]
    high_means = [med_real_high_acc[c] for c in contrasts]

    x = np.arange(len(contrasts))
    width = 0.35

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(
        x - width / 2,
        low_means,
        width,
        label="Low Engagement",
        color="#008BFB",
        alpha=0.9,
        edgecolor="black",
        linewidth=1.2,
    )
    bars2 = plt.bar(
        x + width / 2,
        high_means,
        width,
        label="High Engagement",
        color="#FF8C00",
        alpha=0.9,
        edgecolor="black",
        linewidth=1.2,
    )

    plt.ylabel("Decoding Accuracy", fontsize=12)
    plt.title("Per-Contrast Engagement Decoding Accuracy", fontsize=14, fontweight="bold")
    plt.xticks(x, [f"{c:.4g}" for c in contrasts])
    plt.xlabel("Signed Contrast", fontsize=12)
    plt.legend(frameon=False)
    plt.ylim(0, 1.05)

    # Optional: annotate values
    for bar in bars1:
        yval = bar.get_height()
        if not np.isnan(yval):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.01,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    for bar in bars2:
        yval = bar.get_height()
        if not np.isnan(yval):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.01,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_region(
    eid, region, data_dir, trials_df, all_eids_engagement_signal, base_out_dir, test_type
):
    eid_path = glob.glob(os.path.join(data_dir, "*", eid))
    if not eid_path:
        return None
    eid_path = eid_path[0]

    target_path = os.path.join(eid_path, "targets_real.npy")
    pred_path = os.path.join(eid_path, f"{region}_predictions_real.npy")
    if not (os.path.exists(target_path) and os.path.exists(pred_path)):
        return None

    target = np.load(target_path).squeeze()
    real_predictions = np.load(pred_path).squeeze()

    if eid not in all_eids_engagement_signal:
        return None

    target_engagement_signal = all_eids_engagement_signal[eid]

    has_pseudo = False
    targets_pseudo = None
    predictions_pseudo = None
    N_pseudo = 0
    if test_type in ["hybrid", "pseudo"]:
        pseudo_target_path = os.path.join(eid_path, "targets_pseudo.npy")
        pseudo_pred_path = os.path.join(eid_path, f"{region}_predictions_pseudo.npy")
        if os.path.exists(pseudo_target_path) and os.path.exists(pseudo_pred_path):
            has_pseudo = True
            targets_pseudo = np.load(pseudo_target_path)
            predictions_pseudo = np.load(pseudo_pred_path)
            if targets_pseudo.ndim == 3:
                targets_pseudo = targets_pseudo.squeeze(axis=-1)
            if predictions_pseudo.ndim == 4:
                predictions_pseudo = predictions_pseudo.squeeze(axis=-1)
            N_pseudo = targets_pseudo.shape[0]

        if not has_pseudo:
            return None

    out_dir = os.path.join(base_out_dir, f"{eid}_{region}_{test_type}")
    os.makedirs(out_dir, exist_ok=True)

    res = {"eid": eid, "region": region, "test_type": test_type}

    offset = 1 - target.max()
    corr_test_all = target + offset
    contrasts = sorted(np.unique(corr_test_all[corr_test_all != 0]))

    real_low_accs = {c: [] for c in contrasts}
    real_high_accs = {c: [] for c in contrasts}

    for run_idx in range(real_predictions.shape[0]):
        try:
            a_low, a_high = get_accuracy_engagement(
                target=target,
                pred=real_predictions[run_idx, :],
                trials_df=trials_df,
                engagement_values=target_engagement_signal,
            )
            for c in contrasts:
                if c in a_low:
                    real_low_accs[c].append(a_low[c])
                if c in a_high:
                    real_high_accs[c].append(a_high[c])
        except Exception:
            pass

    med_real_low_acc = {c: np.nanmedian(real_low_accs[c]) for c in contrasts}
    med_real_high_acc = {c: np.nanmedian(real_high_accs[c]) for c in contrasts}

    plot_path = os.path.join(out_dir, "accuracy_grouped_bar.png")
    plot_grouped_accuracy_bar(med_real_low_acc, med_real_high_acc, plot_path)

    if test_type == "none":
        res_rows = []
        for c in contrasts:
            diff = med_real_high_acc[c] - med_real_low_acc[c]
            res_rows.append(
                {
                    "eid": eid,
                    "region": region,
                    "test_type": test_type,
                    "contrast": c,
                    "acc_low": med_real_low_acc[c],
                    "acc_high": med_real_high_acc[c],
                    "delta_acc": diff,
                    "pval_delta_acc": np.nan,
                    "n_nulls": 0,
                }
            )
        res_df = pd.DataFrame(res_rows)
        res_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        return res

    null_low_accs = {c: [] for c in contrasts}
    null_high_accs = {c: [] for c in contrasts}

    if test_type in ["ot", "hybrid"]:
        df_nulls = find_top_sessions(all_eids_engagement_signal, eid, top_n=200)
        for idx in range(min(200, len(df_nulls))):
            source_engagement = all_eids_engagement_signal[df_nulls["session_id"][idx]]
            null_signal = generate_null_interpolation(target_engagement_signal, source_engagement)

            n_la = {c: [] for c in contrasts}
            n_ha = {c: [] for c in contrasts}

            if test_type == "ot":
                for run_idx in range(real_predictions.shape[0]):
                    try:
                        a_low, a_high = get_accuracy_engagement(
                            target=target,
                            pred=real_predictions[run_idx, :],
                            trials_df=trials_df,
                            engagement_values=null_signal,
                        )
                        for c in contrasts:
                            if c in a_low:
                                n_la[c].append(a_low[c])
                            if c in a_high:
                                n_ha[c].append(a_high[c])
                    except Exception:
                        pass
            elif test_type == "hybrid":
                p_idx = idx % N_pseudo
                t_pseudo = targets_pseudo[p_idx, :]
                for run_idx in range(predictions_pseudo.shape[1]):
                    p_pseudo = predictions_pseudo[p_idx, run_idx, :]
                    try:
                        a_low, a_high = get_accuracy_engagement(
                            target=t_pseudo,
                            pred=p_pseudo,
                            trials_df=trials_df,
                            engagement_values=null_signal,
                        )
                        for c in contrasts:
                            if c in a_low:
                                n_la[c].append(a_low[c])
                            if c in a_high:
                                n_ha[c].append(a_high[c])
                    except Exception:
                        pass

            for c in contrasts:
                if n_la[c]:
                    null_low_accs[c].append(np.nanmedian(n_la[c]))
                if n_ha[c]:
                    null_high_accs[c].append(np.nanmedian(n_ha[c]))

    elif test_type == "pseudo":
        for i in range(N_pseudo):
            n_la = {c: [] for c in contrasts}
            n_ha = {c: [] for c in contrasts}

            t_pseudo = targets_pseudo[i, :]
            for run_idx in range(predictions_pseudo.shape[1]):
                p_pseudo = predictions_pseudo[i, run_idx, :]
                try:
                    a_low, a_high = get_accuracy_engagement(
                        target=t_pseudo,
                        pred=p_pseudo,
                        trials_df=trials_df,
                        engagement_values=target_engagement_signal,
                    )
                    for c in contrasts:
                        if c in a_low:
                            n_la[c].append(a_low[c])
                        if c in a_high:
                            n_ha[c].append(a_high[c])
                except Exception:
                    pass
            for c in contrasts:
                if n_la[c]:
                    null_low_accs[c].append(np.nanmedian(n_la[c]))
                if n_ha[c]:
                    null_high_accs[c].append(np.nanmedian(n_ha[c]))

    res_rows = []
    for c in contrasts:
        diff = med_real_high_acc[c] - med_real_low_acc[c]
        if len(null_low_accs[c]) > 0:
            null_diffs = np.array(null_high_accs[c]) - np.array(null_low_accs[c])
            pval = np.mean(null_diffs > diff)
            pval = min(pval, 1 - pval) * 2
            n_nulls = len(null_low_accs[c])
        else:
            pval = np.nan
            n_nulls = 0

        res_rows.append(
            {
                "eid": eid,
                "region": region,
                "test_type": test_type,
                "contrast": c,
                "acc_low": med_real_low_acc[c],
                "acc_high": med_real_high_acc[c],
                "delta_acc": diff,
                "pval_delta_acc": pval,
                "n_nulls": n_nulls,
            }
        )

    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    null_dist_dict = {"low_accs": null_low_accs, "high_accs": null_high_accs}
    with open(os.path.join(out_dir, "null_distributions.pkl"), "wb") as f:
        pkl.dump(null_dist_dict, f)

    return res


def _process_wrapper(row, data_dir, all_eids_engagement_signal, out_dir, test_type):
    eid = row["eid"]
    region = row["region"]
    eid_path = glob.glob(os.path.join(data_dir, "*", eid))
    if not eid_path:
        return None
    trials_path = os.path.join(eid_path[0], "trials.pqt")
    if not os.path.exists(trials_path):
        return None
    trials_df = pd.read_parquet(trials_path)

    print(f"Dispatched {eid} - {region} ({test_type})")
    return process_region(
        eid, region, data_dir, trials_df, all_eids_engagement_signal, out_dir, test_type
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_type", type=str, required=True, choices=["ot", "hybrid", "pseudo", "none"]
    )
    args = parser.parse_args()
    root = "/usr/people/kundu/code/ibl-manifold/"
    local = "/Users/dkundu/Documents/phd/ibl-manifold"
    base_dir = root if os.path.exists(root) else local
    data_dir = os.path.join(base_dir, "data", "ephys_neurometric")
    stage2_path = os.path.join(
        base_dir, "data", "collected_results", "stimulus", "collected_results_stage2.pqt"
    )
    engagement_path = os.path.join(base_dir, "data", "generated", "all_eids_engagement.pkl")
    out_dir = os.path.join(base_dir, "data", "generated", "accuracy_engagement_results")

    os.makedirs(out_dir, exist_ok=True)

    df_stage2 = pd.read_parquet(stage2_path)
    with open(engagement_path, "rb") as f:
        all_eids_engagement_signal = pkl.load(f)

    score_threshold = 0.05
    df_valid = df_stage2[(df_stage2["p-value"] < 0.05) & (df_stage2["score"] > score_threshold)]
    print(f"Found {len(df_valid)} valid eid-region pairs with score > {score_threshold}.")

    print(f"Starting parallel processing for test_type: {args.test_type}")
    results = Parallel(n_jobs=-1)(
        delayed(_process_wrapper)(
            row, data_dir, all_eids_engagement_signal, out_dir, args.test_type
        )
        for idx, row in df_valid.iterrows()
    )

    valid_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_results)} regions.")


if __name__ == "__main__":
    main()
