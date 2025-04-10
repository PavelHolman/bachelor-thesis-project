import logging
import os

import pandas as pd

from src.data.load_all_data import load_all_data
from src.features.generate_day_ahead_features import generate_day_ahead_features
from src.features.generate_intra_day_features import generate_intra_day_features
from src.features.generate_time_features import generate_time_features
from utils import PROCESSED_DATA_DIR


def load_intra_day_dataset(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Load the intraday dataset for the given date range.
    This function loads the data, generates time-based, intraday, and day-ahead features,
    and saves the processed data to a CSV file.
    :param start: Timestamp to start loading data from.
    :param end: Timestamp to stop loading data at.
    :return: DataFrame containing the intraday data.
    """
    logging.info("Loading intraday data")
    df = load_all_data(start, end)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    time_features = generate_time_features(df)
    intra_day_features = generate_intra_day_features(df)
    day_ahead_features = generate_day_ahead_features(df)
    df = pd.concat([df, time_features, intra_day_features, day_ahead_features], axis=1)

    df.to_csv(os.path.join(PROCESSED_DATA_DIR, f"intra_day_{start.date()}_{end.date()}.csv"))
    df["item_id"] = "intra_day"
    df["timestamp"] = df.index
    return df
