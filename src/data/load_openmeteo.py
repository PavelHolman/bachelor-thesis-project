import logging
import os
from typing import Tuple

import pandas as pd

from utils import RAW_DATA_DIR, Inclusive

OPENMETEO_PATH = os.path.join(RAW_DATA_DIR, "openmeteo.csv")


def load_openmeteo_data(start: pd.Timestamp, end: pd.Timestamp, inclusive: Inclusive = "left") -> pd.DataFrame:
    """
    Load Open Meteo raw data. If the data is not available, fetch it from the Open Meteo API.
    :param start: Timestamp to start loading data from.
    :param end: Timestamp to stop loading data at.
    :param inclusive: Whether to include the start and end timestamps in the range.
    :return: DataFrame containing the Open Meteo raw data.
    """
    if os.path.isfile(OPENMETEO_PATH):
        logging.info(f"Loading Open Meteo data from {OPENMETEO_PATH}")
        df = pd.read_csv(OPENMETEO_PATH, parse_dates=["timestamp"], index_col="timestamp")
    else:
        df = pd.DataFrame()

    full_range = pd.date_range(start, end, freq='15min', tz="UTC", inclusive=inclusive)
    missing_range = full_range.difference(df.index)

    if not missing_range.empty:
        missing_start = missing_range.min()
        missing_end = missing_range.max()
        coords = [(lat, long) for lat in range(45, 56, 2) for long in range(5, 24, 2)]
        new_df = __fetch_openmeteo_data(missing_start, missing_end, coords)
        df = pd.concat([df, new_df])
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)

        logging.info(f"Saving Open Meteo data to {OPENMETEO_PATH}")
        df.to_csv(OPENMETEO_PATH)

    return df.loc[full_range]


def __fetch_openmeteo_data(start: pd.Timestamp, end: pd.Timestamp, coords: list[Tuple[float, float]]) -> pd.DataFrame:
    """
    Fetch Open Meteo historical data for the given coordinates and time range.
    :param start: Timestamp to start fetching data from.
    :param end: Timestamp to stop fetching data at.
    :param coords: List of tuples containing latitude and longitude coordinates.
    :return: DataFrame containing the fetched data.
    """
    logging.info(f"Fetching Open Meteo data from {start} to {end} for {len(coords)} coordinates")
    import openmeteo_requests

    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": [lat for (lat, _) in coords],
        "longitude": [long for (_, long) in coords],
        "start_date": start.date(),
        "end_date": end.date(),
        "minutely_15": [
            "temperature_2m", "wind_speed_80m", "rain", "wind_direction_10m", "wind_direction_80m", "wind_speed_10m",
            "wind_gusts_10m", "shortwave_radiation", "direct_radiation", "diffuse_radiation"
        ]
    }
    responses = openmeteo.weather_api(url, params=params)

    dataframes = []

    for response, (lat, long) in zip(responses, coords):
        minutely_15 = response.Minutely15()
        minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy()
        minutely_15_wind_speed_80m = minutely_15.Variables(1).ValuesAsNumpy()
        minutely_15_rain = minutely_15.Variables(2).ValuesAsNumpy()
        minutely_15_wind_direction_10m = minutely_15.Variables(3).ValuesAsNumpy()
        minutely_15_wind_direction_80m = minutely_15.Variables(4).ValuesAsNumpy()
        minutely_15_wind_speed_10m = minutely_15.Variables(5).ValuesAsNumpy()
        minutely_15_wind_gusts_10m = minutely_15.Variables(6).ValuesAsNumpy()
        minutely_15_shortwave_radiation = minutely_15.Variables(7).ValuesAsNumpy()
        minutely_15_direct_radiation = minutely_15.Variables(8).ValuesAsNumpy()
        minutely_15_diffuse_radiation = minutely_15.Variables(9).ValuesAsNumpy()

        minutely_15_data = {
            "timestamp": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left"
            ),
            "Temperature (2m)": minutely_15_temperature_2m,
            "Wind speed (80m)": minutely_15_wind_speed_80m,
            "Rain": minutely_15_rain,
            "Wind direction (10m)": minutely_15_wind_direction_10m,
            "Wind direction (80m)": minutely_15_wind_direction_80m,
            "Wind speed (10m)": minutely_15_wind_speed_10m,
            "Wind gusts (10m)": minutely_15_wind_gusts_10m,
            "Shortwave radiation": minutely_15_shortwave_radiation,
            "Direct radiation": minutely_15_direct_radiation,
            "Diffuse radiation": minutely_15_diffuse_radiation
        }

        minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)
        minutely_15_dataframe.set_index("timestamp", inplace=True)
        minutely_15_dataframe.rename(
            columns=lambda col: col if col == "timestamp" else f"WF ({lat}, {long}) {col}",
            inplace=True)
        dataframes.append(minutely_15_dataframe)

    df = pd.concat(dataframes, axis=1)
    return df
