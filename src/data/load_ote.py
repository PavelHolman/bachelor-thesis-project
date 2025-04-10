import logging
import os
from functools import reduce

import pandas as pd
import requests

from utils import RAW_DATA_DIR

OTE_PATH = os.path.join(RAW_DATA_DIR, "ote.csv")

DAY_AHEAD_URL = "https://www.ote-cr.cz/en/short-term-markets/electricity/day-ahead-market/@@chart-data?report_date={date}"
INTRA_DAY_URL = "https://www.ote-cr.cz/en/short-term-markets/electricity/intra-day-market/@@chart-data?report_date={date}&amp;time_resolution=PT15M"


def load_ote_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Load OTE raw data. If the data is not available, fetch it from the OTE API.
    :param start: Timestamp to start loading data from.
    :param end: Timestamp to stop loading data at.
    :return: DataFrame containing the OTE raw data.
    """
    if os.path.isfile(OTE_PATH):
        logging.info(f"Loading OTE data from {OTE_PATH}")
        df = pd.read_csv(OTE_PATH, parse_dates=["timestamp"], index_col="timestamp")
    else:
        df = pd.DataFrame()

    full_range = pd.date_range(start, end, freq='15min', tz="UTC")
    existing_range = df.index
    missing_range = full_range.difference(existing_range)

    if not missing_range.empty:
        missing_start = missing_range.min()
        missing_end = missing_range.max()
        new_df = __fetch_ote_data(missing_start, missing_end)
        df = pd.concat([df, new_df])
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)

        logging.info(f"Saving OTE data to {OTE_PATH}")
        df.to_csv(OTE_PATH)

    return df.loc[full_range]


def __fetch_ote_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch OTE data for the time range.
    :param start: Timestamp to start fetching data from.
    :param end: Timestamp to stop fetching data at.
    :return: DataFrame containing the fetched data.
    """
    logging.info(f"Fetching OTE data from {start} to {end}")

    intra_day = __fetch_intra_day_data(start, end)
    day_ahead = __fetch_day_ahead_data(start, end)
    df = pd.concat([day_ahead, intra_day], axis=1)
    return df


def __fetch_intra_day_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch intraday data for the time range.
    :param start: Timestamp to start fetching data from.
    :param end: Timestamp to stop fetching data at.
    :return: DataFrame containing the fetched intraday data.
    """
    logging.info(f"Fetching OTE intraday data from {start} to {end}")

    dataframes = []

    for date in pd.date_range(start, end + pd.Timedelta(1, "d"), freq="D"):
        url = INTRA_DAY_URL.format(date=date.date())
        df = _fetch_ote_data_line(url)
        df["timestamp"] = date.tz_convert(None).tz_localize("Europe/Prague") + pd.to_timedelta(15 * df["ts"], unit="m")
        df.drop(columns=["ts"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"
        df.index = df.index.tz_convert("UTC")
        df.rename(columns=lambda col: col if col == "timestamp" else f"ID {col}", inplace=True)
        dataframes.append(df)

    df = pd.concat(dataframes)
    return df


def __fetch_day_ahead_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch day-ahead data for the time range.
    :param start: Timestamp to start fetching data from.
    :param end: Timestamp to stop fetching data at.
    :return: DataFrame containing the fetched day-ahead data.
    """
    logging.info(f"Fetching OTE day-ahead data from {start} to {end}")

    dataframes = []

    for date in pd.date_range(start, end + pd.Timedelta(1, "d"), freq="D"):
        url = DAY_AHEAD_URL.format(date=date.date())
        df = _fetch_ote_data_line(url)
        df["timestamp"] = date.tz_convert(None).tz_localize("Europe/Prague") + pd.to_timedelta(df["ts"], unit="h")
        df.drop(columns=["ts"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"
        df.index = df.index.tz_convert("UTC")
        df.rename(columns=lambda col: col if col == "timestamp" else f"DA {col}", inplace=True)
        dataframes.append(df)

    df = pd.concat(dataframes)
    return df


def _fetch_ote_data_line(url: str) -> pd.DataFrame:
    def transform_data_line(line: dict) -> pd.DataFrame:
        df = pd.DataFrame(line["point"])
        df.rename(columns={"x": "ts", "y": line["title"]}, inplace=True)
        df["ts"] = df["ts"].astype(int) - 1
        return df

    logging.info(f"Fetching OTE data from {url}")
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    data = map(transform_data_line, data["data"]["dataLine"])
    return reduce(lambda left, right: pd.merge(left, right, on="ts"), data)
