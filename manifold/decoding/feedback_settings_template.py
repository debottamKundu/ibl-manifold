import numpy as np
from pathlib import Path

from brainwidemap.decoding.functions.process_targets import optimal_Bayesian
from behavior_models.models import ActionKernel
from sklearn import linear_model as lm

# from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
# from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside
from sklearn import linear_model as lm

# Directory where slurm output and error files will be saved
RESULTS_DIR = Path("./cluster_results")
# BEHFIT_PATH = Path("./results_behavioral")

DATE = "01-04-2023"
# Either current date for a fresh run, or date of the run you want to build on
# Date must be different if you do different runs of the same target
# e.g. signcont side with LogisticRegression vs signcont with Lasso

TARGET = "feedback"
# single-bin targets:
#   'pLeft' - estimate of block prior
#   'signcont' - signed contrast of stimulus
#   'choice' - subject's choice (L/R)
#   'feedback' - correct/incorrect
# multi-bin targets:
#   'wheel-vel' - wheel velocity
#   'wheel-speed' - wheel speed
#   'l-whisker-me' - motion energy of left whisker pad
#   'r-whisker-me' - motion energy of right whisker pad
"""
------------------------------------------------
"""

MODEL = ActionKernel
# behavioral model used for pLeft
# - expSmoothing_prevAction (not string)
# - expSmoothing_stimside (not string)
# - optBay  (not string)
# - oracle (experimenter-defined 0.2/0.8)
# - absolute path; this will be the interindividual results
BEH_MOUSELEVEL_TRAINING = (
    False  # If True, trains the behavioral model session-wise else mouse-wise
)

if TARGET == "feedback":
    ALIGN_TIME = "feedback_times"
    TIME_WINDOW = (0.0, 0.2)
    BINSIZE = 0.2
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
    BINARIZATION_VALUE = 0  # feedback vals are -1 and 1
    TANH_TRANSFORM = False
    EXCLUDE_UNBIASED_TRIALS = False
elif TARGET in ["wheel-vel", "wheel-speed", "l-whisker-me", "r-whisker-me"]:
    ALIGN_TIME = "firstMovement_times"
    TIME_WINDOW = (-0.2, 1.0)
    BINSIZE = 0.02
    N_BINS_LAG = 10
    USE_IMPOSTER_SESSION = True
    BINARIZATION_VALUE = None
    TANH_TRANSFORM = False
    EXCLUDE_UNBIASED_TRIALS = False


# DECODER PARAMS
ESTIMATOR = lm.LogisticRegression
ESTIMATOR_KWARGS = {
    "tol": 0.001,
    "max_iter": 1000,
    "fit_intercept": True,
}  # default args for decoder
if ESTIMATOR == lm.LogisticRegression:
    ESTIMATOR_KWARGS = {**ESTIMATOR_KWARGS, "penalty": "l1", "solver": "liblinear"}
    HPARAM_GRID = {
        "C": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    }  # hyperparameter values to search over
else:
    HPARAM_GRID = {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
N_PSEUDO = 200  # number of pseudo/imposter sessions to fit per session
N_RUNS = 10  # number of times to repeat full nested xv with different folds
SHUFFLE = True  # true for interleaved xv, false for contiguous
BALANCED_WEIGHT = True


# CLUSTER/UNIT PARAMS
MIN_UNITS = 1  # regions with units below this threshold are skipped
SINGLE_REGION = True  # perform decoding on region-wise or whole-brain decoding
MERGED_PROBES = True  # merge probes before performing analysis

# SESSION/BEHAVIOR PARAMS
MIN_BEHAV_TRIAS = (
    1  # minimum number of behavioral trials completed in one session, that fulfill below criteria
)
MIN_RT = 0.08  # remove trials with reaction times above/below these values (seconds), if None, don't apply
MAX_RT = 2.0
MIN_LEN = None  # remove trials with length (feedback_time-goCue_time) above/below these value, if None, don't apply
MAX_LEN = None
