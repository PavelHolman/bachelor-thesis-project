import logging
import os

import pandas as pd

from src.electricity_price_forecasting.features.generate_lag_features import generate_lag_day_ahead_features, \
    generate_rolling_day_ahead_features
from src.electricity_price_forecasting.features.generate_time_features import generate_time_features
from src.electricity_price_forecasting.models.data_split import train_val_test_split
from utils import DAY_AHEAD_MODEL_DIR, PROCESSED_DATA_DIR
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


DA_MODELS = {
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
WEATHER_FEATURES = [f"{lat}_{long}_{col}" for lat in range(45, 56) for long in range(5, 24) for col in [
    "temperature_2m",
    "rain",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_gusts_10m",
    "direct_radiation"]]
WEATHER_FEATURES = [f"{feature} ({q})" for feature in WEATHER_FEATURES for q in [0, 15, 30, 45]]
DA_KNOWN_COV = TIME_FEATURES + WEATHER_FEATURES


def train_day_ahead_models(
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        model_cache: str = "day_ahead",
) -> TimeSeriesPredictor:
    return train_day_ahead_model(train, val, model_cache, known_cov=DA_KNOWN_COV, hyperparameters=DA_MODELS)


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


def train_day_ahead(df: pd.DataFrame) -> tuple[TimeSeriesPredictor, dict[str, float], pd.DataFrame]:
    train, val, test = train_val_test_split(df)
    logging.info(f"Train, val, test split: {train.shape}, {val.shape}, {test.shape}")
    predictor = train_day_ahead_models(train, val)
    results, leaderboard = evaluate_day_ahead_model(predictor, test)
    return predictor, results, leaderboard


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "quarterly.csv"), index_col=0, parse_dates=True)
    df.drop(columns=[col for col in df.columns if "ID" in col], inplace=True)
    generate_lag_day_ahead_features(df)
    generate_rolling_day_ahead_features(df)
    generate_time_features(df)
    df = df.copy()
    train_day_ahead(df)
