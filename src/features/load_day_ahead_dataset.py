import logging
import os

import pandas as pd

from src.data.load_all_data import load_all_data
from src.features.generate_day_ahead_features import generate_day_ahead_features
from src.features.generate_time_features import generate_time_features
from utils import PROCESSED_DATA_DIR


def load_day_ahead_dataset(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Load the day-ahead dataset for the given date range.
    This function loads the data, generates time-based and day-ahead features,
    and saves the processed data to a CSV file.
    Intraday features are not generated for this dataset.
    :param start: Timestamp to start loading data from.
    :param end: Timestamp to stop loading data at.
    :return: DataFrame containing the day-ahead data.
    """
    logging.info("Loading day ahead data")
    df = load_all_data(start, end)
    df.drop(columns=[
        "ID Weighted average of the prices (EUR/MWh)",
        "ID Minimal price(EUR / MWh)",
        "ID Maximal price(EUR / MWh)",
        "ID Volume(MWh)"
    ], inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    time_features = generate_time_features(df)
    day_ahead_features = generate_day_ahead_features(df)
    df = pd.concat([df, time_features, day_ahead_features], axis=1)

    df.to_csv(os.path.join(PROCESSED_DATA_DIR, f"day_ahead_{start.date()}_{end.date()}.csv"))
    df["item_id"] = "day_ahead"
    df["timestamp"] = df.index.tz_convert(None)
    return df
