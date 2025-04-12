import logging
import os

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer

from utils import DAY_AHEAD_MODEL_DIR

__KNOWN = ["T", "WF"]
__MODELS = {}
__TUNING = {}


def train_day_ahead_time_series(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        metric: str | TimeSeriesScorer,
) -> TimeSeriesPredictor:
    path = os.path.join(DAY_AHEAD_MODEL_DIR, "time_series_model")
    try:
        logging.info(f"Loading model from {path}")
        return TimeSeriesPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        return TimeSeriesPredictor(
            target="DA Price (EUR/MWh)",
            known_covariates_names=[col for col in train.columns if any(col.startswith(prefix) for prefix in __KNOWN)],
            freq="h",
            prediction_length=7 * 24,
            eval_metric=metric,
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=__MODELS, hyperparameter_tune_kwargs=__TUNING)
