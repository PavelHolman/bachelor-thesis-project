import logging
import os

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from functools import reduce

from utils import WEATHER_DATA_DIR


def fetch_weather_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dataframes = []

    for latitude in range(45, 56):
        for longitude in range(5, 24):
            data = _get_weather_data(latitude, longitude, start, end)
            data.columns = [f"{latitude}_{longitude}_{col}" for col in data.columns]
            dataframes.append(data)

    return reduce(lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True), dataframes)


def _get_weather_data(latitude: float, longitude: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start, end = start.tz_localize("Europe/Prague"), end.tz_localize("Europe/Prague")
    dataframes = []

    for date in pd.date_range(start, end, inclusive="left"):
        cache_file = os.path.join(WEATHER_DATA_DIR, f"{latitude}_{longitude}_{date.date()}.csv")

        if not os.path.exists(cache_file):
            query_start, query_end = date, min(date + pd.Timedelta(days=30), end)

            logging.info(f"Fetching weather data between {query_start} and {query_end} at {latitude}, {longitude}")
            df = _fetch_weather_data(latitude, longitude, query_start, query_end)

            for query_date in pd.date_range(query_start, query_end, inclusive="left"):
                query_cache_file = os.path.join(WEATHER_DATA_DIR, f"{latitude}_{longitude}_{query_date.date()}.csv")
                query_df = df[df.index.date == query_date.date()]
                if not query_df.empty:
                    query_df.to_csv(query_cache_file)

        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        dataframes.append(df)

    return pd.concat(dataframes)


def _fetch_weather_data(latitude: float, longitude: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.date(),
        "end_date": end.date(),
        "minutely_15": [
            "temperature_2m",
            "rain",
            "wind_speed_10m",
            "wind_speed_80m",
            "wind_direction_10m",
            "wind_direction_80m",
            "wind_gusts_10m",
            "direct_radiation"
        ],
        "timezone": "Europe/Prague"
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

    minutely_15 = response.Minutely15()
    minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy()
    minutely_15_rain = minutely_15.Variables(1).ValuesAsNumpy()
    minutely_15_wind_speed_10m = minutely_15.Variables(2).ValuesAsNumpy()
    minutely_15_wind_speed_80m = minutely_15.Variables(3).ValuesAsNumpy()
    minutely_15_wind_direction_10m = minutely_15.Variables(4).ValuesAsNumpy()
    minutely_15_wind_direction_80m = minutely_15.Variables(5).ValuesAsNumpy()
    minutely_15_wind_gusts_10m = minutely_15.Variables(6).ValuesAsNumpy()
    minutely_15_direct_radiation = minutely_15.Variables(7).ValuesAsNumpy()

    index = pd.date_range(
        start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
        end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=minutely_15.Interval()),
        inclusive="left"
    ).tz_convert("Europe/Prague")

    minutely_15_data = {
        "temperature_2m": minutely_15_temperature_2m,
        "rain": minutely_15_rain,
        "wind_speed_10m": minutely_15_wind_speed_10m,
        "wind_speed_80m": minutely_15_wind_speed_80m,
        "wind_direction_10m": minutely_15_wind_direction_10m,
        "wind_direction_80m": minutely_15_wind_direction_80m,
        "wind_gusts_10m": minutely_15_wind_gusts_10m,
        "direct_radiation": minutely_15_direct_radiation
    }

    return pd.DataFrame(data=minutely_15_data, index=index)
