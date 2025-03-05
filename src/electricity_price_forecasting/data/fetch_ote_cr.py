import os
import logging

import pandas as pd
from pandera.typing import Series
import requests
from functools import reduce
from typing import Optional, Callable

from utils import RAW_DATA_DIR


def fetch_ote_cr_data(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    data_frames: list[pd.DataFrame] = []

    for date in pd.date_range(start=start_date, end=end_date):
        day_ahead_data = fetch_day_ahead_data(date)
        intra_day_data = fetch_intra_day_data(date)
        if day_ahead_data is None or intra_day_data is None:
            continue

        df = pd.merge(day_ahead_data, intra_day_data, on="Datetime", how="outer")
        data_frames.append(df)

    return pd.concat(data_frames)


DAY_AHEAD_URL = "https://www.ote-cr.cz/en/short-term-markets/electricity/day-ahead-market/@@chart-data?report_date={date}"
INTRA_DAY_URL = "https://www.ote-cr.cz/en/short-term-markets/electricity/intra-day-market/@@chart-data?report_date={date}&amp;time_resolution=PT15M"


def fetch_day_ahead_data(date: pd.Timestamp) -> Optional[pd.DataFrame]:
    def time_to_delta(time: Series[int]) -> Series[pd.Timedelta]:
        return pd.to_timedelta(time, unit="h")

    return get_ote_day_data(
        url=DAY_AHEAD_URL,
        prefix="DA",
        date=date,
        time_to_delta=time_to_delta
    )


def fetch_intra_day_data(date: pd.Timestamp) -> Optional[pd.DataFrame]:
    def time_to_delta(time: Series[int]) -> Series[pd.Timedelta]:
        return pd.to_timedelta(15 * time, unit="m")

    return get_ote_day_data(
        url=INTRA_DAY_URL,
        prefix="ID",
        date=date,
        time_to_delta=time_to_delta
    )


def get_ote_day_data(
        url: str,
        prefix: str,
        date: pd.Timestamp,
        time_to_delta: Callable[[Series[int]], Series[pd.Timedelta]]
) -> Optional[pd.DataFrame]:
    path = os.path.join(RAW_DATA_DIR, f"{prefix}_{date.date()}.csv")

    if os.path.exists(path):
        return pd.read_csv(path)

    if (df := fetch_ote_day_data(url, prefix, date, time_to_delta)) is not None:
        if date.date() < pd.Timestamp.now().date():
            df.to_csv(path)

    return df


def fetch_ote_day_data(
        url: str,
        prefix: str,
        date: pd.Timestamp,
        time_to_delta: Callable[[Series[int]], Series[pd.Timedelta]]
) -> Optional[pd.DataFrame]:
    if (df := fetch_ote_data_line(url.format(date=date.date()))) is None:
        return None
    df["Datetime"] = pd.to_datetime(date.strftime("%Y-%m-%d")).tz_localize("Europe/Prague")
    df["Datetime"] += time_to_delta(df["Time"])
    df.set_index("Datetime", inplace=True)
    df.drop(columns=["Time"], inplace=True)
    df.rename(columns={col: f"{prefix} {col}" for col in df.columns if col != "Datetime"}, inplace=True)
    return df


def fetch_ote_data_line(url: str) -> Optional[pd.DataFrame]:
    def transform_data_line(line: dict) -> pd.DataFrame:
        df = pd.DataFrame(line["point"])
        df.rename(columns={"x": "Time", "y": line["title"]}, inplace=True)
        df["Time"] = df["Time"].astype(int) - 1
        return df

    try:
        logging.info(f"Fetching OTE data from {url}")
        response = requests.get(url)
        assert response.status_code == 200
        data = response.json()
        data = map(transform_data_line, data["data"]["dataLine"])
        return reduce(lambda left, right: pd.merge(left, right, on="Time"), data)
    except Exception as e:
        logging.error(f"Failed to fetch OTE data from {url}: {e}")
        return None
