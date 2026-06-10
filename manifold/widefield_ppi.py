# for regions with good enoigh decoding
# we run ppi analysis


import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict


def compute_ppi_interaction(X_stim, Y_choice, labels, alphas=np.logspace(-1, 4, 10), cv=5):
    """
    Computes PPI between two regions

    Parameters:
    -----------
    X_stim : ndarray, shape (n_trials, n_components)
    Y_choice : ndarray, shape (n_trials, n_components)
    labels : ndarray, shape (n_trials,)
        congruent/incongruent
    alphas : array-like
        Regularization parameters to test in RidgeCV.
    cv : int
        Number of cross-validation folds for out-of-sample prediction.

    Returns:
    --------
        r2, betas
    """

    C = np.asarray(labels).reshape(-1, 1)

    Interaction = X_stim * C
    X_full = np.hstack([X_stim, C, Interaction])

    ridge_regression = RidgeCV(alphas=alphas)
    Y_pred_regression = cross_val_predict(ridge_regression, X_full, Y_choice, cv=cv)

    r2_full = r2_score(Y_choice, Y_pred_regression)

    betas_full = ridge_regression.coef_

    n_source_comps = X_stim.shape[1]

    beta_baseline = betas_full[:, :n_source_comps]
    beta_state = betas_full[:, n_source_comps]
    beta_interaction = betas_full[:, n_source_comps + 1 :]

    return r2_full, beta_baseline, beta_interaction
