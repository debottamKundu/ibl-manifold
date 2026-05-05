import logging
import yaml
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import sklearn.linear_model as sklm
from behavior_models.utils import build_path
from .estimators import SoftmaxRegression

logger = logging.getLogger("decoding")


def compute_mask(
    trials_df,
    align_event,
    min_rt=0.08,
    max_rt=None,
    n_trials_crop_end=0,
    keep_timeout_trials=False,
):
    """Create a mask that denotes "good" trials which will be used for further analysis.

    Parameters
    ----------
    trials_df : dict
        contains relevant trial information like goCue_times, firstMovement_times, etc.
    align_event : str
        event in trial on which to align intervals
        'firstMovement_times' | 'stimOn_times' | 'feedback_times'
    min_rt : float
        minimum reaction time; trials with faster reactions will be removed
    max_rt : float
        maximum reaction time; trials with slower reactions will be removed
    n_trials_crop_end : int
        number of trials to crop from the end of the session

    Returns
    -------
    pd.Series
        boolean mask of good trials
    """
    if not isinstance(trials_df, pd.DataFrame):
        if hasattr(trials_df, "to_df"):
            trials_df = trials_df.to_df()
        elif isinstance(trials_df, dict):
            trials_df = pd.DataFrame(trials_df)

    # define reaction times
    react_times = trials_df.firstMovement_times - trials_df.stimOn_times
    # successively build a mask that defines which trials we want to keep

    # ensure align event is not a nan
    mask = trials_df[align_event].notna()

    # ensure animal has moved
    if keep_timeout_trials == False:
        mask = mask & trials_df.firstMovement_times.notna()

        # keep trials with reasonable reaction times
        if min_rt is not None:
            mask = mask & (~(react_times < min_rt)).values
        if max_rt is not None:
            mask = mask & (~(react_times > max_rt)).values

        # get rid of trials where animal does not respond
        mask = mask & (trials_df.choice != 0)

    return mask


def check_inputs(
    model,
    pseudo_ids,
    target,
    output_dir,
    config,
    logger,
    compute_neurometrics=None,
    motor_residuals=None,
):
    """Perform some basic checks and/or corrections on inputs to the main decoding functions"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True)
            logger.info(f"Created output_dir: {output_dir}")
        except PermissionError:
            raise PermissionError(
                f"Following output_dir cannot be created, insufficient permissions: {output_dir}"
            )

    pseudo_ids = [-1] if pseudo_ids is None else pseudo_ids
    if 0 in pseudo_ids:
        raise ValueError(
            "pseudo id can only be -1 (None, actual session) or strictly greater than 0 (pseudo session)"
        )
    if not np.all(np.sort(pseudo_ids) == pseudo_ids):
        raise ValueError("pseudo_ids must be sorted")

    if target in ["choice", "feedback"] and model != "actKernel":
        raise ValueError(
            "If you want to decode choice or feedback, you must use the actionKernel model"
        )

    if compute_neurometrics and target != "signcont":
        raise ValueError(
            "The target should be signcont when compute_neurometrics is set to True in config file"
        )

    if (
        compute_neurometrics
        and len(config["border_quantiles_neurometrics"]) == 0
        and model != "oracle"
    ):
        raise ValueError(
            "If compute_neurometrics is set to True in config file, and model is not oracle, "
            "border_quantiles_neurometrics must be a list of at least length 1"
        )

    if (
        compute_neurometrics
        and len(config["border_quantiles_neurometrics"]) != 0
        and model == "oracle"
    ):
        raise ValueError(
            "If compute_neurometrics is set to True in config file, and model is oracle, "
            "border_quantiles_neurometrics must be set to an empty list"
        )

    if motor_residuals and model != "optBay":
        raise ValueError("Motor residuals can only be computed for optBay model")

    return pseudo_ids, output_dir


def check_config_decoding():
    """Load config yaml and perform some basic checks"""
    # Get config
    config_filename = "config.yml"
    with open(
        Path(__file__).parent.parent.parent.parent.joinpath(config_filename), "r"
    ) as config_yml:
        config = yaml.safe_load(config_yml)
    # Estimator from scikit learn
    try:
        config["estimator"] = getattr(sklm, config["estimator"])
    except AttributeError as e:
        _log = (
            f'The estimator {config["estimator"]} specified in {config_filename} '
            + "is not a function of scikit-learn linear_model."
        )
        try:
            config["estimator"] = globals()[config["estimator"]]
            _log += " Successful local import.\n"
            logger.info(_log)
        except Exception as e:
            _log += " Attempted local import failed!\n"
            logger.error(_log)
            raise e

    # Hyperparameter estimation
    config["use_native_sklearn_for_hyperparam_estimation"] = config["estimator"] == sklm.Ridge
    config["hparam_grid"] = (
        {"C": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
        if config["estimator"] == sklm.LogisticRegression
        else {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])}
    )

    return config


def average_data_in_epoch(
    times, values, trials_df, align_event="stimOn_times", epoch=(-0.6, -0.1)
):
    """
    Aggregate values in a given epoch relative to align_event for each trial. For trials for which the align_event
    is NaN or the epoch contains timestamps outside the times array, the value is set to NaN.

    Parameters
    ----------
    times: np.array
        Timestamps associated with values, assumed to be sorted
    values: np.array
        Data to be aggregated in epoch per trial, one value per timestamp
    trials_df: pd.DataFrame
        Dataframe with trials information
    align_event: str
        Event to align to, must be column in trials_df
    epoch: tuple
        Start and stop of time window to aggregate values in, relative to align_event in seconds


    Returns
    -------
    epoch_array: np.array
        Array of average values in epoch, one per trial in trials_df
    """

    # Make sure timestamps and values are arrays and of same size
    times = np.asarray(times)
    values = np.asarray(values)
    if not len(times) == len(values):
        raise ValueError(
            f"Inputs to times and values must be same length but are {len(times)} and {len(values)}"
        )
    # Make sure times are sorted
    if not np.all(np.diff(times) >= 0):
        raise ValueError("Times must be sorted")
    # Get the events to align to and compute the ideal intervals for each trial
    events = trials_df[align_event].values
    intervals = np.c_[events + epoch[0], events + epoch[1]]
    # Make a mask to exclude trials were the event is nan, or interval starts before or ends after bin_times
    valid_trials = (
        (~np.isnan(events)) & (intervals[:, 0] >= times[0]) & (intervals[:, 1] <= times[-1])
    )
    # This is the first index to include to be sure to get values >= epoch[0]
    epoch_idx_start = np.searchsorted(times, intervals[valid_trials, 0], side="left")
    # This is the first index to exclude (NOT the last to include) to be sure to get values <= epoch[1]
    epoch_idx_stop = np.searchsorted(times, intervals[valid_trials, 1], side="right")
    # Create an array to fill in with the average epoch values for each trial
    epoch_array = np.full(events.shape, np.nan)
    epoch_array[valid_trials] = np.asarray(
        [
            np.nanmean(values[start:stop]) if ~np.all(np.isnan(values[start:stop])) else np.nan
            for start, stop in zip(epoch_idx_start, epoch_idx_stop)
        ],
        dtype=float,
    )

    return epoch_array


def check_bhv_fit_exists(subject, model, eids, resultpath, single_zeta):
    """
    Check if the fit for a given model exists for a given subject and session.

    Parameters
    ----------
    subject: str
        Subject nick name
    model: str
        Model class name
    eids: str or list
        session id or list of session ids for sessions on which model was fitted
    resultpath: str
        Path to the results

    Returns
    -------
    bool
        Whether or not the fit exists
    Path
        Path to the fit
    """
    if isinstance(eids, str):
        eids = [eids]
    fullpath = f"model_{model}"
    if single_zeta:
        fullpath += "_single_zeta"
    fullpath = build_path(fullpath, [eid.split("-")[0] for eid in eids])

    # NOTE: we skip subjects
    fullpath = Path(resultpath).joinpath(fullpath)
    # fullpath = Path(resultpath).joinpath(subject, fullpath), old way
    return fullpath.exists(), fullpath


def downsample_atlas(atlas, pixelSize=20, mask=None):
    return downsampled_atlas.astype(int)


def spatial_down_sample(stack, pixelSize=20):
    return downsampled_im


def subtract_motor_residuals(motor_signals, all_targets, trials_mask):
    """Subtract predictions based on motor signal from predictions as residuals from the behavioural targets"""
    # Update trials mask with possible nans from motor signal
    trials_mask = trials_mask & ~np.any(np.isnan(motor_signals), axis=1)
    # Compute motor predictions and subtract them from targets
    new_targets = []
    for target_data in all_targets:
        clf = sklm.RidgeCV(alphas=[1e-3, 1e-2, 1e-1]).fit(
            motor_signals[trials_mask], target_data[trials_mask]
        )
        motor = np.full_like(target_data, np.nan)
        motor[trials_mask] = clf.predict(motor_signals[trials_mask])
        new_targets.append(target_data - motor)

    return new_targets, trials_mask


def compute_congruency_splits(trials_df, base_mask, n_subsamples=10, seed=42):
    """
    Identifies congruent vs incongruent trials and generates matched subsamples.
    Returns a dictionary of masks to be iterated over.
    """

    is_left_stim = ~np.isnan(trials_df["contrastLeft"])
    is_right_stim = ~np.isnan(trials_df["contrastRight"])
    is_left_block = trials_df["probabilityLeft"] > 0.5
    is_right_block = trials_df["probabilityLeft"] < 0.5

    congruent_mask = (is_left_stim & is_left_block) | (is_right_stim & is_right_block)
    incongruent_mask = (is_left_stim & is_right_block) | (is_right_stim & is_left_block)

    valid_cong = np.where(base_mask & congruent_mask)[0]
    valid_incong = np.where(base_mask & incongruent_mask)[0]

    min_n = min(len(valid_cong), len(valid_incong))
    if min_n == 0:
        raise ValueError("Cannot split: one condition has 0 valid trials.")

    masks = {}
    rng = np.random.default_rng(seed=seed)

    # Incongruent is the minority
    m_i = np.zeros_like(base_mask, dtype=bool)
    m_i[valid_incong] = True
    masks["incongruent"] = m_i

    for i in range(n_subsamples):
        samp_c = rng.choice(valid_cong, min_n, replace=False)
        m_c = np.zeros_like(base_mask, dtype=bool)
        m_c[samp_c] = True
        masks[f"congruent_subsample_{i}"] = m_c

    return masks


def add_congruence(trials):
    out = np.nan_to_num(trials.contrastLeft) - np.nan_to_num(trials.contrastRight)
    left_stim = out > 0
    right_stim = out < 0
    # we throw out 0 contrast trials
    is_left_block = trials["probabilityLeft"] > 0.5  # covers 0.8 blocks
    is_right_block = trials["probabilityLeft"] < 0.5  # covers 0.2 blocks
    congruent = (left_stim & is_left_block) | (right_stim & is_right_block)
    return congruent.values
