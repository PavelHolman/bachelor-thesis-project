import logging
import os

import pandas as pd

from src.data.load_entsoe import load_entsoe_data
from src.data.load_openmeteo import load_openmeteo_data
from src.data.load_ote import load_ote_data
from utils import INTERIM_DATA_DIR, Inclusive

DATA_PATH = os.path.join(INTERIM_DATA_DIR, "data.csv")


def load_all_data(start: pd.Timestamp, end: pd.Timestamp, inclusive: Inclusive = "left") -> pd.DataFrame:
    """
    Load all data sources and merge them into a single DataFrame.
    :param start: Timestamp to start loading data from.
    :param end: Timestamp to stop loading data at.
    :param inclusive: Whether to include the start and end timestamps in the range.
    :return: DataFrame containing the merged data.
    """
    if os.path.isfile(DATA_PATH):
        logging.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")
    else:
        df = pd.DataFrame()

    full_range = pd.date_range(start, end, freq='15min', tz="UTC", inclusive=inclusive)
    missing_range = full_range.difference(df.index)

    if not missing_range.empty:
        missing_start = missing_range.min()
        missing_end = missing_range.max()
        df_entsoe = load_entsoe_data(missing_start, missing_end, inclusive="both")
        df_ote = load_ote_data(missing_start, missing_end, inclusive="both")
        df_openmeteo = load_openmeteo_data(missing_start, missing_end, inclusive="both")

        new_df = pd.concat([df_entsoe, df_ote, df_openmeteo], axis=1)
        df = pd.concat([df, new_df])
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        df.index.name = "timestamp"

        logging.info(f"Saving data to {DATA_PATH}")
        df.to_csv(DATA_PATH)

    return df.loc[full_range]
