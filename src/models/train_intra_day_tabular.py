import logging
import os

import pandas as pd
from autogluon.core.metrics import Scorer
from autogluon.tabular import TabularPredictor

from utils import INTRA_DAY_MODEL_DIR

__MODELS = {
    "RF": {},
    "KNN": {},
    "CAT": {},
    "XGB": {}
}


def train_intra_day_tabular(
        train: pd.DataFrame,
        val: pd.DataFrame,
        metric: str | Scorer,
        cache: str,
        models: dict[str, dict] = None,
) -> TabularPredictor:
    path = os.path.join(INTRA_DAY_MODEL_DIR, f"tabular/{cache}")
    models = models or __MODELS
    try:
        logging.info(f"Loading model from {path}")
        return TabularPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        return TabularPredictor(
            label="ID Weighted average of the prices (EUR/MWh)",
            problem_type="regression",
            eval_metric=metric,
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=models)
