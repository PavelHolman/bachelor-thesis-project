import logging
import os

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer

from src.models.data_split import DA_KNOWN
from utils import DAY_AHEAD_MODEL_DIR

__MODELS = {
    "Naive": {},
    "SeasonalNaive": {},
    "Average": {},
    "SeasonalAverage": {},
    "ETS": {},
    "AutoETS": {},
    "AutoARIMA": {},
    "DirectTabular": {},
    "RecursiveTabular": {},
    "Chronos": {},
}


def train_day_ahead_time_series(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame | None,
        metric: str | TimeSeriesScorer,
        cache: str,
        models: dict[str, dict] = None,
) -> TimeSeriesPredictor:
    """
    Train a time series model for day-ahead price prediction using AutoGluon.
    The prediction length is set to 7 days (168 hours).
    :param train: Training data.
    :param val: Validation data.
    :param metric: Evaluation metric to use for model training.
    :param cache: Cache name for saving the model.
    :param models: Dictionary of models to use for training in AutoGluon format.
    :return: Trained TimeSeriesPredictor model.
    """
    path = os.path.join(DAY_AHEAD_MODEL_DIR, f"time/{cache}")
    models = models or __MODELS
    try:
        logging.info(f"Loading model from {path}")
        return TimeSeriesPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        return TimeSeriesPredictor(
            target="DA Price (EUR/MWh)",
            known_covariates_names=[col for col in train.columns if any(col.startswith(prefix) for prefix in DA_KNOWN)],
            freq="h",
            prediction_length=7 * 24,
            eval_metric=metric,
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=models)
