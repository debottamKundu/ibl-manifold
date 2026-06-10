import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import psychofit as pfit
import ot
import argparse
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from manifold.decoding.functions.neurometric import fit_get_shift_range


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


def get_target_df_engagement(
    target, pred, trials_df, engagement_values=None, engagement_col="engagement"
):
    if engagement_values is not None:
        eng_vals = np.asarray(engagement_values).flatten()
        eng_vals = eng_vals[trials_df["mask"]]
    elif trials_df is not None and engagement_col in trials_df.columns:
        eng_vals = trials_df[engagement_col][trials_df["mask"]].values
    else:
        raise ValueError("Must provide 'engagement_values' or 'trials_df' with 'engagement_col'.")

    median_val = np.nanmedian(eng_vals)
    engagement_bin = eng_vals > median_val

    offset = 1 - target.max()
    corr_test = target + offset
    corr_pred = pred + offset
    pred_signs = np.sign(corr_pred)

    df = pd.DataFrame(
        {
            "stimuli": corr_test,
            "predictions": corr_pred,
            "sign": pred_signs,
            "engagement_bin": engagement_bin,
        }
    )

    grpby = df.groupby(["engagement_bin", "stimuli"])
    grpbyagg = grpby.agg(
        {
            "sign": [
                ("num_trials", "count"),
                ("prop_L", lambda x: ((x == 1).sum() + (x == 0).sum() / 2.0) / len(x)),
            ]
        }
    )

    return [
        grpbyagg.loc[k].reset_index().values.T
        for k in sorted(grpbyagg.index.get_level_values("engagement_bin").unique())
    ]


def plot_linear_engagement_neurometrics(
    x_raw, y_raw_low, y_raw_high, low_curves, high_curves, save_path, region_name
):
    smooth_contrasts = np.linspace(-1.0, 1.0, 100)

    mean_low_curve = np.nanmean(low_curves, axis=0)
    std_low_curve = np.nanstd(low_curves, axis=0)

    mean_high_curve = np.nanmean(high_curves, axis=0)
    std_high_curve = np.nanstd(high_curves, axis=0)

    plt.figure(figsize=(6, 5))

    plt.scatter(x_raw, y_raw_low, color="#008BFB", s=60, alpha=0.8, label="Low Engagement")
    plt.scatter(x_raw, y_raw_high, color="#FF8C00", s=60, alpha=0.8, label="High Engagement")

    plt.plot(smooth_contrasts, mean_low_curve, color="#008BFB", linewidth=3)
    plt.fill_between(
        smooth_contrasts,
        mean_low_curve - std_low_curve,
        mean_low_curve + std_low_curve,
        color="#008BFB",
        alpha=0.2,
    )

    plt.plot(smooth_contrasts, mean_high_curve, color="#FF8C00", linewidth=3)
    plt.fill_between(
        smooth_contrasts,
        mean_high_curve - std_high_curve,
        mean_high_curve + std_high_curve,
        color="#FF8C00",
        alpha=0.2,
    )

    plt.axhline(0.5, color="black", linestyle="--", linewidth=1.5)

    plt.xticks([-1, -0.25, 0, 0.25, 1], labels=["-1", "-0.25", "0", "0.25", "1"])
    plt.xlabel("Signed contrast", fontsize=12)

    plt.ylabel("P(Left)", fontsize=12)
    plt.title(f"Neurometrics - {region_name}", fontsize=14, fontweight="bold")

    plt.ylim(-0.05, 1.05)
    plt.xlim(-1.05, 1.05)
    plt.legend(loc="lower right", frameon=False, fontsize=10)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_region(
    eid,
    region,
    data_dir,
    trials_df,
    all_eids_engagement_signal,
    base_out_dir,
    test_type,
):
    eid_path = glob.glob(os.path.join(data_dir, "*", eid))
    if not eid_path:
        return None
    eid_path = eid_path[0]

    target_path = os.path.join(eid_path, "targets_real.npy")
    pred_path = os.path.join(eid_path, f"{region}_predictions_real.npy")
    if not (os.path.exists(target_path) and os.path.exists(pred_path)):
        return None

    target = np.load(target_path)
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

    smooth_contrasts = np.linspace(-1.0, 1.0, 100)
    real_low_slopes, real_high_slopes, real_shifts = [], [], []
    real_low_curves, real_high_curves = [], []

    for run_idx in range(real_predictions.shape[0]):
        prob_arrs = get_target_df_engagement(
            target=target.squeeze(),
            pred=real_predictions[run_idx, :],
            trials_df=trials_df,
            engagement_values=target_engagement_signal,
        )
        try:
            fit_params = fit_get_shift_range(prob_arrs=prob_arrs)
            real_low_slopes.append(fit_params["low_slope"])
            real_high_slopes.append(fit_params["high_slope"])
            real_shifts.append(fit_params["shift"])

            low_c = pfit.erf_psycho_2gammas(fit_params["low_pars"], smooth_contrasts)
            high_c = pfit.erf_psycho_2gammas(fit_params["high_pars"], smooth_contrasts)
            real_low_curves.append(low_c)
            real_high_curves.append(high_c)
        except Exception:
            pass

    med_real_low_slope = np.nanmedian(real_low_slopes)
    med_real_high_slope = np.nanmedian(real_high_slopes)
    med_real_shift = np.nanmedian(real_shifts)
    med_real_slope_diff = (1.0 / med_real_high_slope) - (1.0 / med_real_low_slope)

    y_pred_mean = np.mean(real_predictions, axis=0)
    prob_arrs_mean = get_target_df_engagement(
        target=target.squeeze(),
        pred=y_pred_mean,
        trials_df=trials_df,
        engagement_values=target_engagement_signal,
    )

    x_raw = prob_arrs_mean[0][0, :]
    y_raw_low = prob_arrs_mean[0][2, :]
    y_raw_high = prob_arrs_mean[-1][2, :]

    plot_path = os.path.join(out_dir, "neurometric.png")
    plot_linear_engagement_neurometrics(
        x_raw,
        y_raw_low,
        y_raw_high,
        np.array(real_low_curves),
        np.array(real_high_curves),
        plot_path,
        region,
    )

    if test_type == "none":
        return 1

    res["med_real_low_slope"] = med_real_low_slope
    res["med_real_high_slope"] = med_real_high_slope
    res["med_real_slope_diff"] = med_real_slope_diff
    res["med_real_shift"] = med_real_shift

    null_low_slopes = []
    null_high_slopes = []
    null_shifts = []

    if test_type in ["ot", "hybrid"]:
        df_nulls = find_top_sessions(all_eids_engagement_signal, eid, top_n=200)
        for idx in range(min(200, len(df_nulls))):
            source_engagement = all_eids_engagement_signal[df_nulls["session_id"][idx]]
            null_signal = generate_null_interpolation(target_engagement_signal, source_engagement)

            n_ls, n_hs, n_sh = [], [], []
            if test_type == "ot":
                for run_idx in range(real_predictions.shape[0]):
                    prob_arrs_null = get_target_df_engagement(
                        target=target.squeeze(),
                        pred=real_predictions[run_idx, :],
                        trials_df=trials_df,
                        engagement_values=null_signal,
                    )
                    try:
                        fit_params_null = fit_get_shift_range(prob_arrs=prob_arrs_null)
                        n_ls.append(fit_params_null["low_slope"])
                        n_hs.append(fit_params_null["high_slope"])
                        n_sh.append(fit_params_null["shift"])
                    except Exception:
                        pass
            elif test_type == "hybrid":
                p_idx = idx % N_pseudo
                t_pseudo = targets_pseudo[p_idx, :]
                for run_idx in range(predictions_pseudo.shape[1]):
                    p_pseudo = predictions_pseudo[p_idx, run_idx, :]
                    try:
                        prob_arrs_hybrid = get_target_df_engagement(
                            target=t_pseudo,
                            pred=p_pseudo,
                            trials_df=trials_df,
                            engagement_values=null_signal,
                        )
                        fit_params_hybrid = fit_get_shift_range(prob_arrs=prob_arrs_hybrid)
                        n_ls.append(fit_params_hybrid["low_slope"])
                        n_hs.append(fit_params_hybrid["high_slope"])
                        n_sh.append(fit_params_hybrid["shift"])
                    except Exception:
                        pass
            if n_ls:
                null_low_slopes.append(np.nanmedian(n_ls))
                null_high_slopes.append(np.nanmedian(n_hs))
                null_shifts.append(np.nanmedian(n_sh))

    elif test_type == "pseudo":
        for i in range(N_pseudo):
            n_ls, n_hs, n_sh = [], [], []
            t_pseudo = targets_pseudo[i, :]
            for run_idx in range(predictions_pseudo.shape[1]):
                p_pseudo = predictions_pseudo[i, run_idx, :]
                try:
                    prob_arrs_pseudo = get_target_df_engagement(
                        target=t_pseudo,
                        pred=p_pseudo,
                        trials_df=trials_df,
                        engagement_values=target_engagement_signal,
                    )
                    fit_params_pseudo = fit_get_shift_range(prob_arrs=prob_arrs_pseudo)
                    n_ls.append(fit_params_pseudo["low_slope"])
                    n_hs.append(fit_params_pseudo["high_slope"])
                    n_sh.append(fit_params_pseudo["shift"])
                except Exception:
                    pass
            if n_ls:
                null_low_slopes.append(np.nanmedian(n_ls))
                null_high_slopes.append(np.nanmedian(n_hs))
                null_shifts.append(np.nanmedian(n_sh))

    if len(null_low_slopes) > 0:
        null_slope_diffs = (1.0 / np.array(null_high_slopes)) - (1.0 / np.array(null_low_slopes))
        p_val_low = np.mean(np.array(null_low_slopes) > res["med_real_low_slope"])
        p_val_high = np.mean(np.array(null_high_slopes) > res["med_real_high_slope"])
        p_val_diff = np.mean(null_slope_diffs > res["med_real_slope_diff"])
        p_val_sh = np.mean(np.array(null_shifts) > res["med_real_shift"])

        res["pval_low_slope"] = min(p_val_low, 1 - p_val_low) * 2
        res["pval_high_slope"] = min(p_val_high, 1 - p_val_high) * 2
        res["pval_slope_diff"] = min(p_val_diff, 1 - p_val_diff) * 2
        res["pval_shift"] = min(p_val_sh, 1 - p_val_sh) * 2
        res["n_nulls"] = len(null_low_slopes)
    else:
        res["pval_low_slope"] = np.nan
        res["pval_high_slope"] = np.nan
        res["pval_slope_diff"] = np.nan
        res["pval_shift"] = np.nan
        res["n_nulls"] = 0

    res_df = pd.DataFrame([res])
    res_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    null_dist_dict = {
        "low_slopes": null_low_slopes,
        "high_slopes": null_high_slopes,
        "shifts": null_shifts,
    }
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

    base_dir = "/Users/dkundu/Documents/phd/ibl-manifold"
    data_dir = os.path.join(base_dir, "data", "ephys_neurometric")
    stage2_path = os.path.join(
        base_dir, "data", "collected_results", "stimulus", "collected_results_stage2.pqt"
    )
    engagement_path = os.path.join(base_dir, "data", "generated", "all_eids_engagement.pkl")
    out_dir = os.path.join(base_dir, "data", "generated", "neurometric_engagement_results")

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
