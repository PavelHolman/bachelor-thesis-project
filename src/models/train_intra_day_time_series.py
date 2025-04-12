import logging
import os

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer

from src.models.data_split import ID_KNOWN
from utils import INTRA_DAY_MODEL_DIR

__MODELS = {}
__TUNING = {}


def train_intra_day_time_series(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        metric: str | TimeSeriesScorer,
) -> TimeSeriesPredictor:
    path = os.path.join(INTRA_DAY_MODEL_DIR, f"time/{metric}")
    try:
        logging.info(f"Loading model from {path}")
        return TimeSeriesPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        return TimeSeriesPredictor(
            target="ID Weighted average of the prices (EUR/MWh)",
            known_covariates_names=[col for col in train.columns if any(col.startswith(prefix) for prefix in ID_KNOWN)],
            freq="15min",
            prediction_length=24 * 4,
            eval_metric=metric,
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=__MODELS, hyperparameter_tune_kwargs=__TUNING)
