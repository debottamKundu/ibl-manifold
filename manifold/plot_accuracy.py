import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_grouped_accuracy_bar_region(df_region, region, save_path):
    # Calculate means and SEMs across sessions for each contrast
    grpby = df_region.groupby("contrast").agg(
        mean_acc_low=("acc_low", "mean"),
        sem_acc_low=("acc_low", "sem"),
        mean_acc_high=("acc_high", "mean"),
        sem_acc_high=("acc_high", "sem"),
        n_sessions=("eid", "count"),
    )

    contrasts = sorted(grpby.index.tolist())
    low_means = grpby.loc[contrasts, "mean_acc_low"].values
    high_means = grpby.loc[contrasts, "mean_acc_high"].values
    low_sems = grpby.loc[contrasts, "sem_acc_low"].values
    high_sems = grpby.loc[contrasts, "sem_acc_high"].values
    n_sess = grpby["n_sessions"].max()

    x = np.arange(len(contrasts))
    width = 0.35

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(
        x - width / 2,
        low_means,
        width,
        yerr=low_sems,
        capsize=4,
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
        yerr=high_sems,
        capsize=4,
        label="High Engagement",
        color="#FF8C00",
        alpha=0.9,
        edgecolor="black",
        linewidth=1.2,
    )

    plt.ylabel("Decoding Accuracy", fontsize=12)
    plt.title(
        f"[{region}] Average Decoding Accuracy (n={n_sess} sessions)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, [f"{c:.4g}" for c in contrasts])
    plt.xlabel("Signed Contrast", fontsize=12)
    plt.legend(frameon=False)
    plt.ylim(0, 1.05)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():

    root = "/usr/people/kundu/code/ibl-manifold/"
    local = "/Users/dkundu/Documents/phd/ibl-manifold"
    base_dir = root

    results_dir = os.path.join(base_dir, "data", "generated", "accuracy_engagement_results")
    out_dir = os.path.join(base_dir, "data", "generated", "accuracy_region_averages")
    os.makedirs(out_dir, exist_ok=True)

    # Recursively find all summary.csv files
    csv_files = glob.glob(os.path.join(results_dir, "*_none", "summary.csv"))

    if len(csv_files) == 0:
        print("No summary.csv files found. Did you run the pipeline with --test_type none?")
        return

    print(f"Found {len(csv_files)} session result files. Aggregating...")

    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception:
            pass

    df_all = pd.concat(df_list, ignore_index=True)

    regions = df_all["region"].unique()
    for region in regions:
        df_region = df_all[df_all["region"] == region]
        plot_path = os.path.join(out_dir, f"{region}_average_accuracy.png")
        plot_grouped_accuracy_bar_region(df_region, region, plot_path)
        print(f"Generated average plot for {region}")

    print(f"\nAll regional average plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
