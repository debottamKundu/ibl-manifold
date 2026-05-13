import logging
import pickle
import numpy as np
import pandas as pd
from brainwidemap.decoding.functions.balancedweightings import balanced_weighting
from brainwidemap.decoding.functions.process_targets import transform_data_for_decoding
from brainwidemap.decoding.functions.process_targets import logisticreg_criteria
from sklearn import linear_model as sklm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import (
    RidgeCV,
    Ridge,
    Lasso,
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.utils.class_weight import compute_sample_weight

from brainbox.io.one import SessionLoader

from .prepare_data import prepare_ephys, prepare_behavior, prepare_motor, prepare_pupil
from .functions.neurometric import get_neurometric_parameters
from .functions.utils import (
    add_congruence,
    check_inputs,
    check_config_decoding,
    compute_mask,
    compute_congruency_splits,
    find_incongruent_trials,
)

# Set up logger
logger = logging.getLogger("decoding_feedback_from_iti")
# Load and check configuration file
config = check_config_decoding()


def fit_session_ephys_feedback(
    one,
    session_id,
    subject,
    pids,
    probe_names,
    output_dir,
    pseudo_ids=None,
    target="feedback",
    align_event="stimOn_times",
    time_window=(-0.4, -0.1),
    model="optBay",
    n_runs=10,
    compute_neurometrics=False,
    motor_residuals=False,
    stage_only=False,
    integration_test=False,
    trials_df=None,
    extra=None,
    behavior_path=None,
    return_ephys=False,
    save_splits=False,
    fit_residuals=None,
    incongruent_only=True,
    balanced_weighting=True,
    trials_mask=None,
):
    """
    Fits a single session for ephys data.
    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    pids: str or list of str
     Probe ID(s), if list of probe IDs, the data of both probes will be merged for decoding
    probe_names: str or list of str
     Probe name(s), corresponding to pids
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    time_window: tuple of float
     Time window in which neural activity is considered, relative to align_event, default is (-0.6, -0.1)
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    compute_neurometrics: bool
     Whether to compute neurometric shift and slopes (cf. Fig 3 of the paper)
    motor_residuals: bool
     Whether ot compute the motor residual before performing neural decoding. This argument is used to study embodiment
     corresponding to figure 2f, default is False
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding
    integration_test: bool
     If true set random seeds for integration testing. Do not use this when running actual decoding
    trials_df: DataFrame or None
     Pandas DataFrame, containing trial information.
    fit_residuals: ... or None
     data series that is used as regressor to give an independent estimate of the target,
     which is then subtracted from the actual target to yield residuals

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(
        model,
        pseudo_ids,
        target,
        output_dir,
        config,
        logger,
        compute_neurometrics,
        motor_residuals,
    )

    # Load trials data and compute mask

    if trials_df is not None:
        assert isinstance(trials_df, pd.DataFrame), "`trials_df` must be a Pandas DataFrame object"
        print("using `trials_df` from kwargs")
    else:
        print(f"ONE in '{one.mode}' mode; ", end="")
        if one.mode == "local":
            print("importing `trials_df` via `One.load_object`")
            trials_df = one.load_object(session_id, "trials")
        else:
            print("importing `trials_df` via `SessionLoader`")
            sl = SessionLoader(one, session_id)
            trials_df = sl.load_trials()
        trials_df = trials_df.to_df()

    # we pass in the mask already
    if trials_mask is None:
        trials_mask = compute_mask(
            trials_df,
            align_event=align_event,
            min_rt=config["min_rt"],
            max_rt=config["max_rt"],
            n_trials_crop_end=0,
            keep_timeout_trials=False,  # keeps all timeout trials,but also fast trials.
        )

    trials_mask = trials_mask.values  # this does use the passed mask.
    if sum(trials_mask) <= config["min_trials"]:
        raise ValueError(
            f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}."
        )

    # for feedback decoding
    if incongruent_only:
        congruent, incongruent = find_incongruent_trials(trials_df)
        trials_mask = incongruent & trials_mask
    else:
        congruent, incongruent = find_incongruent_trials(trials_df)

    # Prepare ephys data
    intervals = np.vstack(
        [trials_df[align_event] + time_window[0], trials_df[align_event] + time_window[1]]
    ).T
    data_epoch, actual_regions = prepare_ephys(
        one,
        session_id,
        pids,
        probe_names,
        config["regions"],  # this will essentially run it over the entire IBL dataset.
        intervals,
        qc=config["unit_qc"],
        min_units=config["min_units"],
        stage_only=stage_only,
    )
    if stage_only:
        print(f"Staged {session_id}")
    if not len(actual_regions):
        print("No decoding")
        return
    else:
        print(f"Decoding from {', '.join(['_'.join(r) for r in actual_regions])}")

    # Compute or load behavior targets
    all_trials, all_targets, trials_mask, pseudo_congruence = prepare_behavior(
        session_id,
        subject,
        trials_df,
        trials_mask,
        pseudo_ids=pseudo_ids,
        output_dir=output_dir,
        model=model,
        target=target,
        compute_neurometrics=compute_neurometrics,
        integration_test=integration_test,
        behavior_path=behavior_path,
        incongruent_only=incongruent_only,
    )

    trials_df["mask"] = trials_mask
    trials_df.to_parquet(output_dir.joinpath(subject, session_id, "trials.pqt"))

    if stage_only:
        return

    # Create strings for saving
    pseudo_str = f"{pseudo_ids[0]}_{pseudo_ids[-1]}" if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    if isinstance(probe_names, list):
        if len(probe_names) > 1:
            probe_str = "merged_probes"
        else:
            probe_str = probe_names[0]
    else:
        probe_str = probe_names

    session_dir = output_dir.joinpath(subject, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    pseudo_targets = []
    all_targets = [np.reshape(t, (-1, 1)) if not len(t[0].shape) else t for t in all_targets]
    # no mask if not incongruent_only
    if incongruent_only:
        final_masks = []
        for p_id, p_mask in zip(pseudo_ids, pseudo_congruence):  # type: ignore

            if isinstance(p_mask, list) and len(p_mask) == 2:
                raise ValueError("should not happem, incongruent only")
                m = p_mask[1]  # Extract the incongruent mask
            else:
                m = p_mask

            m = np.squeeze(np.array(m)).astype(bool)
            if p_id == -1:
                final_masks.append(trials_mask & m)
            else:
                final_masks.append(m)

        actual_nb_trials = 0

        # 2. Save targets and congruence
        for i, (t, p_id) in enumerate(zip(all_targets, pseudo_ids)):
            mask = final_masks[i]
            if p_id == -1:
                np.save(session_dir.joinpath("targets_real.npy"), t[mask])
                # Store the final trial count to use in the output dict
                actual_nb_trials = mask.sum()
            else:
                pseudo_targets.append(t[mask])
                m_orig = pseudo_congruence[i]  # type: ignore
                if isinstance(m_orig, list) and len(m_orig) == 2:
                    m_arr = np.array(m_orig[1])
                else:
                    m_arr = np.array(m_orig)
                pseudo_congruence_array.append(m_arr[mask])

    for i, (t, p_id) in enumerate(zip(all_targets, pseudo_ids)):
        if p_id == -1:
            np.save(session_dir.joinpath("targets_real.npy"), t[trials_mask])
            # Store the final trial count to use in the output dict
            actual_nb_trials = trials_mask.sum()
        else:
            pseudo_targets.append(t[trials_mask])
            m_orig = pseudo_congruence[i]  # type: ignore

    n_pseudo = len(pseudo_targets)
    if n_pseudo:
        np.save(session_dir.joinpath("targets_pseudo.npy"), np.stack(pseudo_targets))
        np.save(session_dir.joinpath("congruence_pseudo.npy"), np.stack(pseudo_congruence))  # type: ignore

    # Otherwise fit per region
    filenames = []
    for data_region, region in zip(data_epoch, actual_regions):

        # append the extra variables to the input data
        data = data_region

        print(f"\n  Decoding from {len(data_region[0])} neurons in {'_'.join(region)}")
        if incongruent_only:
            neural_mask = final_masks[0]

            fit_results = fit_target(
                data[neural_mask],  # type: ignore
                [t[m] for t, m in zip(all_targets, final_masks)],  # U
                all_trials,
                n_runs,
                None,  # all neurometrics
                pseudo_ids,
                integration_test=integration_test,
                save_splits=save_splits,
                balanced_weighting=balanced_weighting,
            )
        else:
            fit_results = fit_target(
                data[trials_mask],  # type: ignore
                [t[trials_mask] for t in all_targets],  # U
                all_trials,
                n_runs,
                None,  # all neurometrics
                pseudo_ids,
                integration_test=integration_test,
                save_splits=save_splits,
                balanced_weighting=balanced_weighting,
            )

        # Create output paths and save
        region_str = f"{'_'.join(region)}"

        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "probe": probe_str,
            "region": region,
            "N_units": data_region.shape[1],
        }

        # save the predictions for the actual session
        # (n_runs, nb_trials)
        _pred = np.stack(
            [res["predictions_test"] for res in fit_results if res["pseudo_id"] == -1]
        )
        np.save(session_dir.joinpath(f"{region_str}_predictions_real.npy"), _pred)
        _shape = _pred[0].shape

        # save the predictions for the pseudo sessions
        # (n_pseudo, n_runs, nb_trials)
        if n_pseudo:
            _pred = np.stack(
                [res["predictions_test"] for res in fit_results if res["pseudo_id"] != -1]
            )
            _pred = _pred.reshape(n_pseudo, n_runs, *_shape)
            np.save(session_dir.joinpath(f"{region_str}_predictions_pseudo.npy"), _pred)

        # for each decoding (session, region) produce some summary of the results
        # including R2 scores and weights
        filename = session_dir.joinpath(f"{region_str}_decoding_summary.pqt")
        _df = make_summary_df(outdict)
        _df.to_parquet(filename)
        filenames.append(filename)

    if return_ephys:
        return filenames, list(zip(actual_regions, data_epoch))
    else:
        return filenames


def make_summary_df(outdict):
    fit_results = outdict["fit"]
    n_results = len(fit_results)
    _weights = [
        np.mean(np.vstack(res["weights"]), axis=0) for res in fit_results
    ]  # mean weights over folds
    _intercept = [
        np.mean(np.vstack(res["intercepts"]).squeeze(), axis=0) for res in fit_results
    ]  # mean intercept over folds
    _dict = dict(
        eid=[outdict["eid"] for _ in range(n_results)],
        condition=[outdict.get("condition", "all") for _ in range(n_results)],
        subject=[outdict["subject"] for _ in range(n_results)],
        probe=[outdict["probe"] for _ in range(n_results)],
        region=["_".join(outdict["region"]) for _ in range(n_results)],
        pseudo_id=[res["pseudo_id"] for res in fit_results],
        run_id=[res["run_id"] for res in fit_results],
        N_units=_weights[0].shape[-1],
        R2_test=[res["scores_test_full"] for res in fit_results],
        nb_trials=[len(res["target"]) for res in fit_results],
        weights=_weights,
        intercept=_intercept,
    )
    df = pd.DataFrame(_dict)
    return df


def fit_target(
    data_to_fit,
    all_targets,
    all_trials,
    n_runs,
    all_neurometrics=None,
    pseudo_ids=None,
    integration_test=False,
    save_splits=False,
    balanced_weighting=True,
):
    """
    Fits data (neural, motor, etc) to behavior targets.

    Parameters
    ----------
    data_to_fit : list of np.ndarray
        List of neural or other data, each element is a (n_trials, n_units) array with the averaged neural activity
    all_targets : list of np.ndarray
        List of behavior targets, each element is a (n_trials,) array with the behavior targets for one (pseudo)session
    all_trials : list of pd.DataFrames
        List of trial information, each element is a pd.DataFrame with the trial information for one (pseudo)session
    n_runs: int
        Number of times to repeat full nested cross validation with different folds
    all_neurometrics : list of pd.DataFrames or None
        List of neurometrics, each element is a pd.DataFrame with the neurometrics for one (pseudo)session.
        If None, don't compute neurometrics. Default is None
    pseudo_ids : list of int or None
        List of pseudo session ids, -1 indicates the actual session. If None, run only on actual session.
        Default is None.
    integration_test : bool
        Whether to run in integration test mode with fixed random seeds. Default is False.
    """

    # Loop over (pseudo) sessions and then over runs
    if pseudo_ids is None:
        pseudo_ids = [-1]
    if not all_neurometrics:
        all_neurometrics = [None] * len(all_targets)
    fit_results = []
    for targets, trials, neurometrics, pseudo_id in zip(
        all_targets, all_trials, all_neurometrics, pseudo_ids
    ):
        n_runs_null = 2  # speed up run
        current_n_runs = (
            n_runs if pseudo_id == -1 else n_runs_null
        )  # NOTE: This reduces the number of runs for
        # run decoders
        for i_run in range(current_n_runs):
            rng_seed = i_run if integration_test else None
            fit_result = decode_cv(
                ys=targets,
                Xs=data_to_fit,
                use_openturns=False,
                estimator=config["estimator"],
                estimator_kwargs=config["estimator_kwargs"],
                hyperparam_grid=config["hparam_grid"],
                save_binned=False,
                save_predictions=config["save_predictions"],
                shuffle=config["shuffle"],
                balanced_weight=balanced_weighting,
                rng_seed=rng_seed,
            )

            fit_result["trials_df"] = trials
            fit_result["pseudo_id"] = pseudo_id
            fit_result["run_id"] = i_run

            fit_results.append(fit_result)

    return fit_results


def decode_cv(
    ys,
    Xs,
    estimator,
    estimator_kwargs,
    use_openturns,
    target_distribution=None,
    balanced_continuous_target=False,
    balanced_weight=False,
    hyperparam_grid=None,
    test_prop=0.2,
    n_folds=5,
    save_binned=False,
    save_predictions=True,
    verbose=False,
    shuffle=True,
    outer_cv=True,
    rng_seed=None,
    normalize_input=False,
    normalize_output=False,
):
    """Regresses binned neural activity against a target, using a provided sklearn estimator.

    Parameters
    ----------
    ys : list of arrays or np.ndarray or pandas.Series
        targets; if list, each entry is an array of targets for one trial. if 1D numpy array, each
        entry is treated as a single scalar for one trial. if pd.Series, trial number is the index
        and teh value is the target.
    Xs : list of arrays or np.ndarray
        predictors; if list, each entry is an array of neural activity for one trial. if 2D numpy
        array, each row is treated as a single vector of ativity for one trial, i.e. the array is
        of shape (n_trials, n_neurons)
    estimator : sklearn.linear_model object
        estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
    estimator_kwargs : dict
        additional arguments for sklearn estimator
    use_openturns : bool
    target_distribution : ?
        ?
    balanced_weight : ?
        ?
    balanced_continuous_target : ?
        ?
    hyperparam_grid : dict
        key indicates hyperparameter to grid search over, and value is an array of nodes on the
        grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
        Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
    test_prop : float
        proportion of data to hold out as the test set after running hyperparameter tuning; only
        used if `outer_cv=False`
    n_folds : int
        Number of folds for cross-validation during hyperparameter tuning; only used if
        `outer_cv=True`
    save_binned : bool
        True to put the regressors Xs into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    save_predictions : bool
        True to put the model predictions into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    shuffle : bool
        True for interleaved cross-validation, False for contiguous blocks
    outer_cv: bool
        Perform outer cross validation such that the testing spans the entire dataset
    rng_seed : int
        control data splits
    verbose : bool
        Whether you want to hear about the function's life, how things are going, and what the
        neighbor down the street said to it the other day.
    normalize_output : bool
        True to take out the mean across trials of the output
    normalize_input : bool
        True to take out the mean across trials of the input; average is taken across trials for
        each unit (one average per unit is computed)

    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Decoding intercept
            - Per-trial target values (copy of tvec)
            - Per-trial predictions from model
            - Input regressors (optional, see Xs argument)

    """

    # transform target data into standard format: list of np.ndarrays
    # print(Xs.shape)
    # print(ys.shape)
    # ys = np.where(ys == -1, 0, ys)
    print("shape is")
    print(ys.shape)
    ys, Xs = transform_data_for_decoding(ys, Xs)
    # print(len(Xs))
    # print(len(ys))
    # print(ys)
    # ys = np.ravel(ys)
    # print(ys)

    # initialize containers to save outputs
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression

    # split the dataset in two parts, train and test
    # when shuffle=False, the method will take the end of the dataset to create the test set
    if rng_seed is not None:
        np.random.seed(rng_seed)
    # create indicies to loop over
    indices = np.arange(n_trials)
    if outer_cv:
        # create kfold function to loop over
        get_kfold = lambda: KFold(n_splits=n_folds, shuffle=shuffle).split(indices)
        # define function to evaluate whether folds are satisfactory
        if estimator == sklm.LogisticRegression:
            # folds must be chosen such that 2 classes are present in each fold
            assert logisticreg_criteria(ys)
            isysat = lambda ys: logisticreg_criteria(
                ys, MIN_UNIQUE_COUNTS=2
            )  # len(np.unique(ys))==2 and np.min(np.unique(ys ,return_counts=True)[1])>=2
        else:
            isysat = lambda ys: True
        sample_count, _, outer_fold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1:
            print(f"sampled outer folds {sample_count} times to ensure enough targets")

        # old way of getting non logistic regression folds. now incorporated into above
        # else:
        #    outer_fold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(get_kfold())]

    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_fold_iter = [
            (train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)
        ]

    # scoring function; use R2 for linear regression, accuracy for logistic regression
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if (
        estimator == sklm.RidgeCV
        or estimator == sklm.LassoCV
        or estimator == sklm.LogisticRegressionCV
    ):
        raise NotImplementedError("the code does not support a CV-type estimator for the moment.")
    else:
        # loop over outer folds
        for train_idxs_outer, test_idxs_outer in outer_fold_iter:
            # outer fold data split
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            # create indicies and kfold function to loop over inner folds
            idx_inner = np.arange(len(X_train))
            get_kfold = lambda: KFold(n_splits=n_folds, shuffle=shuffle).split(idx_inner)

            # produce inner_fold_iter such that logistic regression has at least 2 classes
            if estimator == sklm.LogisticRegression:
                # is it possible to construct folds
                y_uniquecounts = np.unique(y_train, return_counts=True)[1]
                assert logisticreg_criteria(
                    y_train, MIN_UNIQUE_COUNTS=2
                )  # len(y_uniquecounts)==2 and np.min(y_uniquecounts)>=2 #print('failed inner fold, target unique counts:', y_uniquecounts)

                # folds must be chosen such that 2 classes are present in each fold
                isysat = lambda ys: logisticreg_criteria(
                    ys, MIN_UNIQUE_COUNTS=1
                )  # len(np.unique(ys))==2 and np.min(np.unique(ys ,return_counts=True)[1])>=1
            else:
                isysat = lambda ys: True
            sample_count, _, inner_fold_iter = sample_folds(y_train, get_kfold, isysat)
            if sample_count > 1:
                print(f"sampled inner folds {sample_count} times to ensure enough targets")

            key = list(hyperparam_grid.keys())[0]  # type: ignore # TODO: make this more robust
            r2s = np.zeros([n_folds, len(hyperparam_grid[key])])  # type: ignore
            for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_fold_iter):

                # inner fold data split
                X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                # normalize inputs/outputs if requested
                mean_X_train_inner = X_train_inner.mean(axis=0) if normalize_input else 0
                X_train_inner = X_train_inner - mean_X_train_inner
                X_test_inner = X_test_inner - mean_X_train_inner
                mean_y_train_inner = y_train_inner.mean(axis=0) if normalize_output else 0
                y_train_inner = y_train_inner - mean_y_train_inner

                for i_alpha, alpha in enumerate(hyperparam_grid[key]):  # type: ignore

                    # compute weight for each training sample if requested
                    # (esp necessary for classification problems with imbalanced classes)
                    if balanced_weight:
                        sample_weight = balanced_weighting(
                            vec=y_train_inner,
                            continuous=balanced_continuous_target,
                            use_openturns=use_openturns,
                            target_distribution=target_distribution,
                        )
                    else:
                        sample_weight = None

                    # initialize model
                    model_inner = estimator(**{**estimator_kwargs, key: alpha})
                    # fit model
                    model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                    # evaluate model
                    pred_test_inner = model_inner.predict(X_test_inner) + mean_y_train_inner
                    r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

            # select model with best hyperparameter value evaluated on inner-fold test data;
            # refit/evaluate on all inner-fold data
            r2s_avg = r2s.mean(axis=0)

            # normalize inputs/outputs if requested
            X_train_array = np.vstack(X_train)
            mean_X_train = X_train_array.mean(axis=0) if normalize_input else 0
            X_train_array = X_train_array - mean_X_train

            y_train_array = np.concatenate(y_train, axis=0)
            mean_y_train = y_train_array.mean(axis=0) if normalize_output else 0
            y_train_array = y_train_array - mean_y_train

            # compute weight for each training sample if requested
            if balanced_weight:
                sample_weight = balanced_weighting(
                    vec=y_train_array,
                    continuous=balanced_continuous_target,
                    use_openturns=use_openturns,
                    target_distribution=target_distribution,
                )
            else:
                sample_weight = None

            # initialize model
            best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]  # type: ignore
            model = estimator(**{**estimator_kwargs, key: best_alpha})
            # fit model
            model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            # evalute model on train data
            y_pred_train = model.predict(X_train_array) + mean_y_train
            scores_train.append(
                scoring_f(y_train_array + mean_y_train, y_pred_train + mean_y_train)
            )

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)
            y_pred = model.predict(np.vstack(X_test) - mean_X_train) + mean_y_train
            if isinstance(model, sklm.LogisticRegression) and bins_per_trial == 1:
                # print("predicting proba in decoding of logistic regression!")
                y_pred_probs = (
                    model.predict_proba(np.vstack(X_test) - mean_X_train)[:, 1] + mean_y_train
                )
                # print(f"example of proba: {y_pred_probs[0]:.5f}")
                # print(y_pred_probs[:100], y_pred[:100])
                # print(np.isclose(y_pred_probs[:100], y_pred[:100]))
                y_comp_probs = ~(y_pred_probs == 0.5)
                # print(y_pred_probs[y_comp_probs] > 0.5)
                # print(y_pred[y_comp_probs])
                # assert np.all((y_pred_probs[y_comp_probs] > 0.5) == y_pred[y_comp_probs])
            else:
                print(
                    "did not predict proba",
                    estimator,
                    isinstance(model, sklm.LogisticRegression),
                    bins_per_trial,
                )
                y_pred_probs = None
            scores_test.append(scoring_f(y_true, y_pred))

            # save the raw prediction in the case of linear and the predicted probabilities when
            # working with logitistic regression
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    # we already computed these estimates, take from above
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])  # type: ignore
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    # we already computed these above, but after all trials were stacked;
                    # recompute per-trial
                    predictions[i_global] = (
                        model.predict(X_test[i_fold] - mean_X_train) + mean_y_train
                    )
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = (
                            model.predict_proba(X_test[i_fold] - mean_X_train)[:, 0] + mean_y_train
                        )
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # save out other data of interest
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:  # type: ignore
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    ys_true_full = np.concatenate(ys, axis=0)
    ys_pred_full = np.concatenate(predictions, axis=0)  # type: ignore
    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)
    if estimator == sklm.LogisticRegression:
        outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
        outdict["balanced_acc_test_full"] = balanced_accuracy_score(ys_true_full, ys_pred_full)
    outdict["weights"] = weights
    outdict["intercepts"] = intercepts
    outdict["target"] = ys
    outdict["predictions_test"] = predictions_to_save if save_predictions else None
    outdict["regressors"] = Xs if save_binned else None
    outdict["idxes_test"] = idxes_test
    outdict["idxes_train"] = idxes_train
    outdict["best_params"] = best_params
    outdict["n_folds"] = n_folds
    if hasattr(model, "classes_"):
        outdict["classes_"] = model.classes_

    # logging
    if verbose:
        # verbose output
        if outer_cv:
            print("Performance is only described for last outer fold \n")
        print("Possible regularization parameters over {} validation sets:".format(n_folds))
        print("{}: {}".format(list(hyperparam_grid.keys())[0], hyperparam_grid))
        print("\nBest parameters found over {} validation sets:".format(n_folds))
        print(model.best_params_)
        print("\nAverage scores over {} validation sets:".format(n_folds))
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, model.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n", "Detailed scores on {} validation sets:".format(n_folds))
        for i_fold in range(n_folds):
            tscore_fold = list(
                np.round(model.cv_results_["split{}_test_score".format(int(i_fold))], 3)
            )
            print("perf on fold {}: {}".format(int(i_fold), tscore_fold))

        print("\n", "Detailed classification report:", "\n")
        print("The model is trained on the full (train + validation) set.")

    return outdict


def sample_folds(ys, get_kfold, isfoldsat, max_iter=100):
    sample_count = 0
    ysatisfy = [False]
    while not np.all(np.array(ysatisfy)):
        assert sample_count < max_iter
        sample_count += 1
        out_kfold = get_kfold()
        fold_iter = [
            (train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(out_kfold)
        ]
        ysatisfy = [
            isfoldsat(np.concatenate([ys[i] for i in t_idxs], axis=0)) for t_idxs, _ in fold_iter
        ]

    return sample_count, out_kfold, fold_iter
