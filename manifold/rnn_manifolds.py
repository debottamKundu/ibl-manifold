from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns


def get_category_centroids(data, combine_contrasts=False):
    """
    Calculates the mean hidden state (centroid) for each of the 8 trial categories.
    """
    results = {}

    if combine_contrasts:
        # Concatenate arrays across all contrasts
        combined_data = {
            "correct": np.concatenate([np.asarray(data[c]["correct"]) for c in data]),
            "trial_side": np.concatenate([np.asarray(data[c]["trial_side"]) for c in data]),
            "block_side": np.concatenate([np.asarray(data[c]["block_side"]) for c in data]),
            "hidden_state_seq": np.concatenate(
                [np.asarray(data[c]["hidden_state_seq"]) for c in data], axis=0
            ),
        }
        results["all_contrasts"] = _compute_centroids(combined_data)
    else:
        # Process each contrast individually
        for contrast, contrast_data in data.items():
            results[contrast] = _compute_centroids(contrast_data)

    return results


def _compute_centroids(contrast_data):
    """Helper function to apply logical masks and compute the mean hidden state."""

    correct = contrast_data["correct"] == 1
    left_stim = contrast_data["trial_side"] == -1
    left_block = contrast_data["block_side"] == -1

    conditions = {
        "ls_lb_correct": left_stim & left_block & correct,
        "ls_rb_correct": left_stim & ~left_block & correct,
        "rs_lb_correct": ~left_stim & left_block & correct,
        "rs_rb_correct": ~left_stim & ~left_block & correct,
        "ls_lb_incorrect": left_stim & left_block & ~correct,
        "ls_rb_incorrect": left_stim & ~left_block & ~correct,
        "rs_lb_incorrect": ~left_stim & left_block & ~correct,
        "rs_rb_incorrect": ~left_stim & ~left_block & ~correct,
    }

    hs_full = contrast_data["hidden_state_seq"][:, 0, :].squeeze()

    centroids = {}
    for name, mask in conditions.items():
        # Check if there are any trials for this condition to avoid Mean of Empty Slice warnings
        if np.any(mask):
            # Calculate the mean across the trials dimension (axis 0)
            centroids[name] = np.mean(hs_full[mask], axis=0)
        else:
            centroids[name] = None  # No trials for this specific condition

    return centroids


def plot_centroids(centroids_dict, contrast_key="all_contrasts"):
    """
    Plots the 3D centroids of the hidden states for a specific contrast.
    """
    if contrast_key not in centroids_dict:
        print(f"Error: '{contrast_key}' not found. Available keys: {list(centroids_dict.keys())}")
        return

    centroids = centroids_dict[contrast_key]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {"ls_lb": "blue", "ls_rb": "cyan", "rs_lb": "red", "rs_rb": "magenta"}

    for condition, point in centroids.items():
        if point is None:
            continue  # Skip if no data for this condition

        base_cond = condition.replace("_correct", "").replace("_incorrect", "")
        color = color_map.get(base_cond, "gray")

        is_correct = "incorrect" not in condition
        marker = "o" if is_correct else "X"

        ax.scatter(
            point[0],
            point[1],
            point[2],
            c=color,
            marker=marker,
            s=200,
            edgecolors="black",
            label=condition,
        )

        # Add a text label slightly offset from the point
        label_text = condition.replace("_", " ")
        ax.text(point[0] + 0.05, point[1] + 0.05, point[2] + 0.05, label_text, fontsize=9)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.8, 0.8)

    ax.set_title(f"Category Centroids in 3D Hidden State Space ({contrast_key})", fontsize=14)
    ax.set_xlabel("Hidden State 1")
    ax.set_ylabel("Hidden State 2")
    ax.set_zlabel("Hidden State 3")

    # Legend mapping
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Conditions")
    # plt.tight_layout()
    plt.show()


def plot_centroids_subplots(centroids_dict):
    """
    Plots the 3D centroids for each contrast on a separate subplot.

    Args:
        centroids_dict (dict): The output from get_category_centroids(data, combine_contrasts=False).
    """

    n_contrasts = len(centroids_dict)
    if n_contrasts == 0:
        print("Error: The centroids dictionary is empty.")
        return

    cols = int(np.ceil(np.sqrt(n_contrasts)))
    rows = int(np.ceil(n_contrasts / cols))

    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    color_map = {"ls_lb": "blue", "ls_rb": "cyan", "rs_lb": "red", "rs_rb": "magenta"}

    handles, labels = [], []

    for i, (contrast_name, centroids) in enumerate(centroids_dict.items(), 1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        ax.set_title(f"Contrast: {contrast_name}", fontsize=12)

        for condition, point in centroids.items():
            if point is None:
                continue

            base_cond = condition.replace("_correct", "").replace("_incorrect", "")
            color = color_map.get(base_cond, "gray")

            is_correct = "incorrect" not in condition
            marker = "o" if is_correct else "X"

            scatter = ax.scatter(
                point[0],
                point[1],
                point[2],
                c=color,
                marker=marker,
                s=150,
                edgecolors="black",
                label=condition,
            )

            # Add short text labels to avoid overcrowding the subplots
            short_label = condition.replace("_correct", " (C)").replace("_incorrect", " (I)")
            ax.text(point[0], point[1], point[2] + 0.05, short_label, fontsize=8)

            # Collect handles for the global legend (only from the first subplot)
            if i == 1:
                handles.append(scatter)
                labels.append(condition)

            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 0.8)
            ax.set_zlim(-0.8, 0.8)
        # Basic axis labels for each subplot
        ax.set_xlabel("HS 1")
        ax.set_ylabel("HS 2")
        ax.set_zlabel("HS 3")

    # Add a single global legend to the right of the entire figure
    if handles:
        fig.legend(
            handles, labels, loc="center right", title="Conditions", bbox_to_anchor=(1.1, 0.5)
        )

    plt.tight_layout()
    plt.show()


import numpy as np
from sklearn.decomposition import PCA


def get_2d_pca_centroids(data):
    """
    Fits a global 2D PCA across all trials, computes condition centroids,
    and projects them into the shared 2D space.
    """

    all_hidden_states = []
    for contrast, contrast_data in data.items():
        hs = np.asarray(contrast_data["hidden_state_seq"])

        hs_2d = hs.reshape(hs.shape[0], -1)
        all_hidden_states.append(hs_2d)

    combined_hs = np.vstack(all_hidden_states)

    pca = PCA(n_components=2)
    pca.fit(combined_hs)
    explained_var = pca.explained_variance_ratio_

    results = {}
    for contrast, contrast_data in data.items():
        correct = np.asarray(contrast_data["correct"]) == 1
        left_stim = np.asarray(contrast_data["trial_side"]) == -1
        left_block = np.asarray(contrast_data["block_side"]) == -1

        conditions = {
            "ls_lb_correct": left_stim & left_block & correct,
            "ls_rb_correct": left_stim & ~left_block & correct,
            "rs_lb_correct": ~left_stim & left_block & correct,
            "rs_rb_correct": ~left_stim & ~left_block & correct,
            "ls_lb_incorrect": left_stim & left_block & ~correct,
            "ls_rb_incorrect": left_stim & ~left_block & ~correct,
            "rs_lb_incorrect": ~left_stim & left_block & ~correct,
            "rs_rb_incorrect": ~left_stim & ~left_block & ~correct,
        }

        hs_full = np.asarray(contrast_data["hidden_state_seq"])
        hs_full_2d = hs_full.reshape(hs_full.shape[0], -1)

        centroids_2d = {}
        for name, mask in conditions.items():
            if np.any(mask):

                centroid_3d = np.mean(hs_full_2d[mask], axis=0)
                centroid_2d = pca.transform(centroid_3d.reshape(1, -1))[0]
                centroids_2d[name] = centroid_2d
            else:
                centroids_2d[name] = None

        results[contrast] = centroids_2d

    return results, explained_var


def plot_2d_centroids_subplots(centroids_dict, explained_var):
    """
    Plots the 2D PCA-projected centroids for each contrast on a separate subplot.
    """
    n_contrasts = len(centroids_dict)
    if n_contrasts == 0:
        print("Error: The centroids dictionary is empty.")
        return

    cols = int(np.ceil(np.sqrt(n_contrasts)))
    rows = int(np.ceil(n_contrasts / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharex=True, sharey=True)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    color_map = {"ls_lb": "blue", "ls_rb": "cyan", "rs_lb": "red", "rs_rb": "magenta"}

    handles, labels = [], []

    for i, ((contrast_name, centroids), ax) in enumerate(zip(centroids_dict.items(), axes)):
        ax.set_title(f"Contrast: {contrast_name}", fontsize=12)

        for condition, point in centroids.items():
            if point is None:
                continue

            base_cond = condition.replace("_correct", "").replace("_incorrect", "")
            color = color_map.get(base_cond, "gray")

            is_correct = "incorrect" not in condition
            is_congruent = base_cond in ["ls_lb", "rs_rb"]

            marker = "o" if is_correct else "X"

            if is_congruent:
                face_color = color  # Solid fill
                edge_color = "black"  # Standard black border
                line_width = 1.0
            else:
                face_color = "white"  # Hollow inside
                edge_color = color  # The border becomes the condition color
                line_width = 2.5  # Make the colored border thicker so it's easy to see

            scatter = ax.scatter(
                point[0],
                point[1],
                facecolors=face_color,
                edgecolors=edge_color,
                linewidths=line_width,
                marker=marker,
                s=150,
                label=condition,
            )

            # Add short text labels slightly offset
            short_label = condition.replace("_correct", " (C)").replace("_incorrect", " (I)")
            ax.text(point[0] + 0.02, point[1] + 0.02, short_label, fontsize=12)

            # 3. Fixed the index check (enumerate with zip starts at 0)
            if i == 0:
                handles.append(scatter)
                labels.append(condition)

        ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Clean up empty subplots
    for j in range(len(centroids_dict), len(axes)):
        axes[j].set_visible(False)

    if handles:
        fig.legend(
            handles, labels, loc="center right", title="Conditions", bbox_to_anchor=(1.1, 0.5)
        )

    plt.tight_layout()
    sns.despine()
    plt.show()


def predict_choice_from_start_state(data):
    """
    Trains a classifier to predict the final choice from the t=0 hidden state.
    """
    all_hs = []
    all_choices = []

    for contrast, contrast_data in data.items():
        # Get starting states (t=0) and flatten extra dimensions
        hs_start = np.asarray(contrast_data["hidden_state_seq"])[:, 0, :]
        hs_start = hs_start.reshape(hs_start.shape[0], -1)

        choices = contrast_data["action_side"]

        all_hs.append(hs_start)
        all_choices.append(choices)

    X = np.vstack(all_hs)
    y = np.concatenate(all_choices)

    clf = LogisticRegression()
    scores = cross_val_score(clf, X, y, cv=5)

    print(f"Mean Choice Prediction Accuracy from t=0: {np.mean(scores)*100:.1f}%")
    return clf, X, y
