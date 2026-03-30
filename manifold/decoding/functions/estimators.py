import pandas as pd
import numpy as np
from scipy.optimize import minimize, curve_fit

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import euclidean_distances
from scipy.special import softmax, log_softmax


class SoftmaxRegression(BaseEstimator, RegressorMixin):
    """
    Like LogisticRegression, but fits a multivariate categorical distribution
    instead of the categorically distributed data.

    """

    _noise_init = 0.1

    def __init__(
        self,
        method=None,
        fit_intercept=True,
        alpha=0.01,
        tol=None,
        max_iter=20000,
    ):
        super().__init__()
        self.method = method
        self.fit_intercept = fit_intercept
        self._alpha = alpha
        self._tol = tol
        self._maxiter = max_iter

    def _init_params(self, X, y):

        # Initialize the weights first.
        # If there are more than 2 classes, class 1 is chosen as reference,
        # so that its weights are set to be 0,  and the weights for class 2
        # are interpreted as "weights relative to class 1"
        self._n_weights = self.n_classes - int(self.n_classes == 2)
        self._params = np.random.randn(self._n_weights, self.n_features) / np.sqrt(self.n_features)

        if self.fit_intercept:
            _bias = np.atleast_1d(np.random.randn(self._n_weights))
            self._params = np.vstack([self._params.T, _bias]).T
        self._params += self._noise_init * np.random.randn(*self._params.shape)

    def _check_data(self, X, y):
        if not len(y.shape) == 2:
            raise ValueError(
                "Target must be an array (n_samples, n_classes), with n_classes >= 2."
            )

        if not len(y) == len(X):
            raise ValueError("Mismatch in the number of inputs and targets.")

        _norm = np.sum(y, axis=1)
        if not (np.allclose(_norm, 1) and np.all(y >= 0)):
            raise ValueError("Target must be a probability mass function.")

        self.n_samples, self.n_features = X.shape
        _, self.n_classes = y.shape
        self._n_weights = self.n_classes - int(self.n_classes == 2)

    def fit(self, X, y, sample_weight=None):

        self._check_data(X, y)

        self._init_params(X, y)
        _params_init = np.reshape(self._params, (-1,))

        self._X = X
        self._y = y

        opt = minimize(self._f, _params_init)  # , tol=self._tol)
        self._params = opt["x"].reshape(self._n_weights, -1)

        return self

    def predict(self, X):

        # Input validation
        # e.g. if X is a pandas.DataFrame
        # it returns X.to_numpy()
        X = check_array(X)

        return self._model(X, self._params)

    @property
    def coef_(self):
        _weights = self._get_weights(self._params)
        return _weights[-self._n_weights :].copy()

    @property
    def intercept_(self):
        _biases = self._get_biases(self._params)
        return _biases[-self._n_weights :].copy()

    def _get_weights(self, params):
        if self.fit_intercept:
            _weights = params[:, :-1]
        else:
            _weights = params

        # If there are only 2 classes, class 1 is chosen as reference,
        # so that its weights are set to be 0, and the weights for class 2
        # are interpreted as "weights relative to class 1"
        _temp = np.zeros((self.n_classes, self.n_features))
        _temp[-self._n_weights :] = _weights
        _weights = _temp

        return _weights

    def _get_biases(self, params):
        if self.n_classes == 2:
            self._n_weights = 1
        else:
            self._n_weights = self.n_classes

        _biases = np.zeros(self.n_classes)
        if self.fit_intercept:
            _biases[-self._n_weights :] = params[:, -1]

        return _biases

    def _linear_function(self, X, params):
        _params = np.reshape(params, self._params.shape)
        _weights = self._get_weights(_params)
        _biases = self._get_biases(_params)
        return X @ _weights.T + _biases

    def _crossentropy_loss(self, y_obs, log_y_pred):
        return -np.mean(np.sum(y_obs * log_y_pred, axis=1))

    def _log_model(self, x, params):
        _z = self._linear_function(x, params)
        return log_softmax(_z, axis=1)

    def _model(self, x, params):
        _z = self._linear_function(x, params)
        return softmax(_z, axis=1)

    # function to be minimized
    def _f(self, params, *args):
        x = self._X
        y_obs = self._y
        log_y_pred = self._log_model(x, params)
        # position of arguments matters!!!
        l = self._crossentropy_loss(y_obs, log_y_pred)
        l += 0.5 * self._alpha * np.sum(params**2)
        return l


if __name__ == "__main__":

    np.random.seed(1971)

    NUM_POINTS = 300  # number of points
    INPUT_DIM = 2  # number of regressors
    OUTPUT_DIM = 3  # number of classes
    LABEL_NOISE = 0.0

    if OUTPUT_DIM == 1:
        raise ValueError("Output dimension must be at least 2.")
    else:
        w0 = 2 * np.eye(max(INPUT_DIM, OUTPUT_DIM))[:OUTPUT_DIM, :INPUT_DIM]
        if OUTPUT_DIM == 2:
            w0 = w0[1:]
        else:
            w0 *= np.arange(1, OUTPUT_DIM + 1)[::-1, None]

    def f_target(X):
        if OUTPUT_DIM == 2:
            _w0 = np.vstack([np.zeros(w0.shape[1]), w0])
        else:
            _w0 = w0
        z = X @ _w0.T
        return softmax(z, axis=1)

    X = np.random.randn(NUM_POINTS, INPUT_DIM)
    y = f_target(X)
    y = y * (1 + LABEL_NOISE * np.random.randn(NUM_POINTS, OUTPUT_DIM))
    y[y < 0] = 0
    y = y / np.sum(y, axis=1)[:, None]

    print("Check normalization:", np.all(np.sum(y, axis=1)))

    classes = np.array([np.random.choice(OUTPUT_DIM, p=p) for p in y])
    print("Sampled classes: ", np.unique(classes))

    alpha = 0.0001
    fit_intercept = True
    print("target\n", w0, "\n===================")

    log = LogisticRegression(C=0.5 / alpha, fit_intercept=fit_intercept)
    log.fit(X, classes)
    print("logistic result\n", log.coef_, "\n", log.intercept_, "\n===================")

    sig = SoftmaxRegression(alpha=alpha, fit_intercept=fit_intercept)
    sig.fit(X, y)
    print("softmax result\n", sig.coef_, "\n", sig.intercept_, "\n===================")
    predicted_sig = sig.predict(X)
    print(f"R^2 softmax = {sig.score(X,y)}")

    predicted_log = log.predict(X)
    print(f"accuracy logistic  = {log.score(X,classes)}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.tight_layout()
    im = ax.scatter(*X.T, c=classes, cmap="magma", edgecolors=(0, 0, 0))
    for i, (wSig, wLog, wTrue) in enumerate(zip(sig.coef_, log.coef_, w0)):
        ax.arrow(0, 0, *wTrue, color=f"C{i}", ls="-", lw=1)
        ax.arrow(0, 0, *wSig, color=f"C{i}", ls="--", lw=2)
        ax.arrow(0, 0, *wLog, color=f"C{i}", ls=":", lw=3)
    fig.colorbar(im, ax=ax)
    fig.savefig("classes.png", dpi=300)

    fig, ax = plt.subplots()
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    for i, (_x, _y) in enumerate(zip(y.T, predicted_sig.T)):
        ax.scatter(_x, _y, edgecolors=(0, 0, 0), c=classes, cmap="magma")
    fig.savefig("true-vs-predicted.png", dpi=300)
