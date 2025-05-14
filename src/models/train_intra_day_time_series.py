import logging
import os

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer

from src.testing.data_split import ID_KNOWN, ID_TARGET
from utils import INTRA_DAY_MODEL_DIR

__MODELS = {
    "Naive": {},
    "SeasonalNaive": [{"seasonal_period": 24}, {"seasonal_period": 7 * 24}],
    "Average": {},
    "SeasonalAverage": [{"seasonal_period": 24}, {"seasonal_period": 7 * 24}],
    "AutoETS": [{"seasonal_period": 24}, {"seasonal_period": 7 * 24}],
    "AutoARIMA": [{"seasonal_period": 24}, {"seasonal_period": 7 * 24}],
    "DirectTabular": {},
    "RecursiveTabular": {},
    "Chronos": {},
}


def train_intra_day_time_series(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame | None,
        metric: str | TimeSeriesScorer,
        cache: str,
        models: dict[str, dict] = None,
) -> TimeSeriesPredictor:
    """
    Train a time series model for intraday price prediction using AutoGluon.
    The prediction length is set to 1 day (96 * 15-minutes).
    :param train: Training data.
    :param val: Validation data.
    :param metric: Evaluation metric to use for model training.
    :param cache: Cache name for saving the model.
    :param models: Dictionary of models to use for training in AutoGluon format.
    :return: Trained TimeSeriesPredictor model.
    """
    path = os.path.join(INTRA_DAY_MODEL_DIR, f"time/{cache}")
    models = models or __MODELS
    try:
        logging.info(f"Loading model from {path}")
        predictor = TimeSeriesPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        predictor = TimeSeriesPredictor(
            target=ID_TARGET,
            known_covariates_names=[col for col in train.columns if any(col.startswith(prefix) for prefix in ID_KNOWN)],
            freq="15min",
            prediction_length=24 * 4,
            eval_metric=metric,
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=models)

    logging.debug(f"{predictor.info()}")
    logging.debug(f"{predictor.fit_summary()}")
    return predictor
