import logging
import os

import pandas as pd
from data_split import train_val_test_split
from src.electricity_price_forecasting.features.full_intraday import full_intraday
from utils import INTRADAY_MODEL_DIR
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_intraday_model(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        model_cache: str,
        known_cov: list[str] = None,
        hyperparameters: dict = None
) -> TimeSeriesPredictor:
    path = os.path.join(INTRADAY_MODEL_DIR, model_cache)
    try:
        logging.info(f"Loading model from {path}")
        return TimeSeriesPredictor.load(path)
    except FileNotFoundError:
        logging.info(f"Model not found, training a new model")
        return TimeSeriesPredictor(
            target="ID Weighted average of the prices (EUR/MWh)",
            known_covariates_names=[x for x in known_cov if x in train.columns] if known_cov else None,
            freq="15min",
            prediction_length=24 * 4,
            eval_metric="MAE",
            path=path,
            verbosity=2,
        ).fit(train, val, hyperparameters=hyperparameters)


ID_MODELS = {
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

TIME_FEATURES = ["hour_of_day", "day_of_week", "month", "year", "is_holiday"]
POWER_GRID_FEATURES = ["Forecasted Load", "Actual Aggregated", "Actual Consumption", "Solar"]
DA_FEATURES = (["DA Price (EUR/MWh)", "DA Volume (MWh)"] +
               [f"da_lag_{lag}" for lag in range(1, 7 * 24 + 1)] +
               [f"da_rolling_mean_{window}" for window in range(24, 7 * 24 + 1, 24)] +
               [f"da_rolling_std_{window}" for window in range(24, 7 * 24 + 1, 24)])
WEATHER_FEATURES = [f"{lat}_{long}_{col}" for lat in range(45, 56) for long in range(5, 24) for col in [
    "temperature_2m",
    "rain",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_gusts_10m",
    "direct_radiation"]]
ID_KNOWN_COV = TIME_FEATURES + POWER_GRID_FEATURES + DA_FEATURES + WEATHER_FEATURES


def train_intraday_models(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        model_cache: str = "intraday",
) -> TimeSeriesPredictor:
    return train_intraday_model(train, val, model_cache, known_cov=ID_KNOWN_COV, hyperparameters=ID_MODELS)


def evaluate_day_ahead_model(
        predictor: TimeSeriesPredictor,
        test: TimeSeriesDataFrame,
        metrics: list[str] = None,
        display: bool = True
) -> tuple[dict[str, float], pd.DataFrame]:
    if metrics is None:
        metrics = ["MAE", "RMSE", "MAPE"]
    results = predictor.evaluate(test, metrics=metrics, display=display)
    leaderboard = predictor.leaderboard(test, display=display)
    return results, leaderboard


def train_intraday(df: pd.DataFrame) -> tuple[TimeSeriesPredictor, dict[str, float], pd.DataFrame]:
    train, val, test = train_val_test_split(df)
    logging.info(f"Train, val, test split: {train.shape}, {val.shape}, {test.shape}")
    predictor = train_intraday_models(train, val)
    results, leaderboard = evaluate_day_ahead_model(predictor, test)
    return predictor, results, leaderboard


if __name__ == "__main__":
    df = full_intraday()
    train_intraday(df)
