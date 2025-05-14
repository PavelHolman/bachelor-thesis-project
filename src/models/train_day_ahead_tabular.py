import logging
import os

import pandas as pd
from autogluon.core.metrics import Scorer
from autogluon.tabular import TabularPredictor

from src.testing.data_split import DA_TARGET
from utils import DAY_AHEAD_MODEL_DIR

__MODELS = {
    "RF": {},
    "KNN": {},
    "CAT": {},
    "XGB": {}
}


def train_day_ahead_tabular(
        train: pd.DataFrame,
        val: pd.DataFrame,
        metric: str | Scorer,
        cache: str,
        models: dict[str, dict] = None,
) -> TabularPredictor:
    """
    Train a tabular model for day-ahead price prediction using AutoGluon.
    :param train: Training data.
    :param val: Validation data.
    :param metric: Evaluation metric to use for model training.
    :param cache: Cache name for saving the model.
    :param models: Dictionary of models to use for training in AutoGluon format.
    :return: Trained TabularPredictor model.
    """
    path = os.path.join(DAY_AHEAD_MODEL_DIR, f"tabular/{cache}")
    models = models or __MODELS
    try:
        logging.info(f"Loading model from {path}")
        predictor = TabularPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        predictor = TabularPredictor(
            label=DA_TARGET,
            problem_type="regression",
            eval_metric=metric,
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=models)

    logging.debug(f"{predictor.info()}")
    logging.debug(f"{predictor.fit_summary()}")
    return predictor
