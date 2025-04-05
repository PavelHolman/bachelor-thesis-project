import logging
import os

import pandas as pd
from data_split import train_val_test_split
from utils import PROCESSED_DATA_DIR, DAY_AHEAD_MODEL_DIR
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_day_ahead_model(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        model_cache: str,
        known_cov: list[str] = None,
        hyperparameters: dict = None
) -> TimeSeriesPredictor:
    path = os.path.join(DAY_AHEAD_MODEL_DIR, model_cache)
    try:
        logging.info(f"Loading model from {path}")
        return TimeSeriesPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        return TimeSeriesPredictor(
            target="DA Price (EUR/MWh)",
            known_covariates_names=known_cov,
            freq="h",
            prediction_length=7 * 24,
            eval_metric="MAE",
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=hyperparameters)


def main():
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "merged.csv"), index_col=0, parse_dates=True)
    train, val, test = train_val_test_split(df)
    logging.info(f"Train, val, test split: {train.shape}, {val.shape}, {test.shape}")
    predictor = train_day_ahead_model(train, val, "merged")
    predictor.evaluate(test, metrics=["MAE", "RMSE", "MAPE"], display=True)
    predictor.leaderboard(test, display=True)


if __name__ == "__main__":
    main()
