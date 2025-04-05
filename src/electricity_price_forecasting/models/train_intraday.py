import logging
import os

import pandas as pd

from data_split import train_val_test_split
from utils import MODEL_DIR, PROCESSED_DATA_DIR
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, \
    root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_intraday_model(df: pd.DataFrame, model_cache: str):
    df["item_id"] = "intraday"
    df["timestamp"] = df.index.tz_convert(None)

    train, val, test = train_val_test_split(df)
    train = TimeSeriesDataFrame.from_data_frame(train)
    val = TimeSeriesDataFrame.from_data_frame(val)
    test = TimeSeriesDataFrame.from_data_frame(test)
    logging.info(f"Train, val, test split: {train.shape}, {val.shape}, {test.shape}")

    try:
        logging.info(f"Loading model from {os.path.join(MODEL_DIR, model_cache)}")
        predictor = TimeSeriesPredictor.load(os.path.join(MODEL_DIR, model_cache))
    except FileNotFoundError:
        logging.info(f"Training model from {os.path.join(MODEL_DIR, model_cache)}")
        predictor = TimeSeriesPredictor(
            target="ID Weighted average of the prices (EUR/MWh)",
            # TODO known_covariates_names=[],
            freq="15min",
            prediction_length=24 * 4,
            eval_metric="MAPE",
            path=os.path.join(MODEL_DIR, model_cache),
            verbosity=2,
        ).fit(train, val)

    y_pred, y_test = [], []
    i = 0
    logging.info(f"Predicting for {len(test)} rows")
    while True:
        try:
            i_test = df.iloc[len(train) + len(val) + 24 * 4 * (i - 7):len(train) + len(val) + 24 * 4 * i]

            y_pred.append(predictor.predict(i_test))
            y_test.append(i_test["ID Weighted average of the prices (EUR/MWh)"])
            i += 1
        except IndexError:
            break

    y_test = [y for l in y_test for y in l]
    y_pred = [y for d in y_pred for y in d['mean']]
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Root Mean Squared Error: {rmse}")

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Mean Absolute Percentage Error: {100 * mape}%")

    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2}")
    predictor.leaderboard(test)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "merged.csv"), index_col=0, parse_dates=True)
    train_intraday_model(df, "intraday")
