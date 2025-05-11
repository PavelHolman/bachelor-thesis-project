import logging
import os

import pandas as pd

from src.data.load_all_data import load_all_data
from src.features.generate_day_ahead_features import generate_day_ahead_features
from src.features.generate_time_features import generate_time_features
from utils import PROCESSED_DATA_DIR


def load_day_ahead_dataset(start: pd.Timestamp, end: pd.Timestamp, save: bool=False) -> pd.DataFrame:
    """
    Load the day-ahead dataset for the given date range.
    This function loads the data, generates time-based and day-ahead features,
    and saves the processed data to a CSV file.
    Intraday features are not generated for this dataset.
    :param start: Timestamp to start loading data from.
    :param end: Timestamp to stop loading data at.
    :param save: Whether to save the processed data to a CSV file.
    :return: DataFrame containing the day-ahead data.
    """
    logging.info("Loading day ahead data")
    df = load_all_data(start, end)
    df.drop(columns=[col for col in df.columns if col.startswith("ID")], inplace=True)
    df = unfold_15_minutely_columns(df)
    time_features = generate_time_features(df)
    day_ahead_features = generate_day_ahead_features(df)
    df = pd.concat([df, time_features, day_ahead_features], axis=1).dropna()

    if save:
        df.to_csv(os.path.join(PROCESSED_DATA_DIR, f"day_ahead_{start.date()}_{end.date()}.csv"))
    return df


def unfold_15_minutely_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    sparse_columns = set(col for col in df.columns if df[col].isna().sum() != 0)
    unfolded_parts = []

    for col in df.columns:
        if col in sparse_columns:
            unfolded_parts.append(df[col])
            continue
        for i in range(4):
            shifted = df[col].shift(-i)
            shifted.name = f"{col} (T+{i * 15})"
            unfolded_parts.append(shifted)

    unfolded_df = pd.concat(unfolded_parts, axis=1)
    result = unfolded_df.iloc[::4]

    result.index.name = df.index.name or "timestamp"
    return result
