import logging
import os.path
import pandas as pd
from entsoe import EntsoePandasClient
from typing import Callable
from utils import ENTSOE_API_KEY, LOAD_FORECAST_DIR, WIND_AND_SOLAR_FORECAST_DIR, GENERATION_FORECAST_DIR


def fetch_generation_forecast(country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return _fetch_query(
        country_code=country_code,
        start=start,
        end=end,
        cache_dir=GENERATION_FORECAST_DIR,
        query_api_call=lambda code, s, e: _client.query_generation_forecast(country_code=code, start=s, end=e)
    )


def fetch_wind_and_solar_forecast(country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return _fetch_query(
        country_code=country_code,
        start=start,
        end=end,
        cache_dir=WIND_AND_SOLAR_FORECAST_DIR,
        query_api_call=lambda code, s, e: _client.query_wind_and_solar_forecast(country_code=code, start=s, end=e)
    )


def fetch_load_forecast(country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return _fetch_query(
        country_code=country_code,
        start=start,
        end=end,
        cache_dir=LOAD_FORECAST_DIR,
        query_api_call=lambda code, s, e: _client.query_load_forecast(country_code=code, start=s, end=e)
    )


_client = EntsoePandasClient(api_key=ENTSOE_API_KEY)


def _fetch_query(
        country_code: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cache_dir: str,
        query_api_call: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame],
) -> pd.DataFrame:
    start, end = start.tz_localize("Europe/Prague"), end.tz_localize("Europe/Prague")
    dataframes = []

    for date in pd.date_range(start, end, inclusive="left"):
        cache_file = os.path.join(cache_dir, f"{country_code}_{date.date()}.csv")

        if not os.path.exists(cache_file):
            query_start, query_end = date, date + pd.Timedelta(days=30)
            try:
                logging.info(f"Fetching data for {country_code} between {query_start} and {query_end}")
                df = query_api_call(country_code, query_start, query_end)
            except Exception as e:
                logging.error(f"Failed to fetch data for {country_code} between {query_start} and {query_end}: {e}")
                continue

            for query_date in pd.date_range(query_start, query_end, inclusive="left"):
                query_cache_file = os.path.join(cache_dir, f"{country_code}_{query_date.date()}.csv")
                query_df = df[df.index.date == query_date.date()]
                if not query_df.empty:
                    query_df.to_csv(query_cache_file)

        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        dataframes.append(df)

    return pd.concat(dataframes)
