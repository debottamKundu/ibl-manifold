import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_grouped_accuracy_bar_region(df_region, region, save_path):
    # Calculate means across sessions for each contrast
    grpby = df_region.groupby("contrast").agg(
        mean_acc_low=("acc_low", "mean"),
        mean_acc_high=("acc_high", "mean"),
        n_sessions=("eid", "count"),
    )

    contrasts = sorted(grpby.index.tolist())
    low_means = grpby.loc[contrasts, "mean_acc_low"].values
    high_means = grpby.loc[contrasts, "mean_acc_high"].values
    n_sess = grpby["n_sessions"].max()

    x = np.arange(len(contrasts))
    width = 0.35

    plt.figure(figsize=(12, 6))

    # Plot transparent bars
    plt.bar(
        x - width / 2,
        low_means,
        width,
        label="Low Engagement (Mean)",
        color="#008BFB",
        alpha=0.3,
        edgecolor="black",
        linewidth=1.2,
    )
    plt.bar(
        x + width / 2,
        high_means,
        width,
        label="High Engagement (Mean)",
        color="#FF8C00",
        alpha=0.3,
        edgecolor="black",
        linewidth=1.2,
    )

    # Plot individual paired points connected by lines
    for i, c in enumerate(contrasts):
        df_c = df_region[df_region["contrast"] == c].dropna(subset=["acc_low", "acc_high"])
        x_low = x[i] - width / 2
        x_high = x[i] + width / 2

        for _, row in df_c.iterrows():
            y_low = row["acc_low"]
            y_high = row["acc_high"]

            # Draw line
            plt.plot(
                [x_low, x_high], [y_low, y_high], color="gray", alpha=0.5, linewidth=1.0, zorder=2
            )
            # Draw points
            plt.scatter(
                x_low,
                y_low,
                color="#008BFB",
                s=25,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            plt.scatter(
                x_high,
                y_high,
                color="#FF8C00",
                s=25,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    plt.ylabel("Decoding Accuracy", fontsize=12)
    plt.title(
        f"[{region}] Average Decoding Accuracy (n={n_sess} sessions)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, [f"{c:.4g}" for c in contrasts])
    plt.xlabel("Signed Contrast", fontsize=12)
    plt.legend(frameon=False, loc="lower right")
    plt.ylim(0, 1.05)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():

    root = "/usr/people/kundu/code/ibl-manifold/"
    local = "/Users/dkundu/Documents/phd/ibl-manifold"
    base_dir = root if os.path.exists(root) else local

    results_dir = os.path.join(base_dir, "data", "generated", "accuracy_engagement_results")
    out_dir = os.path.join(local, "data", "generated", "accuracy_region_averages")
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
