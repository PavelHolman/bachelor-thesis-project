import logging
import os

import pandas as pd
from entsoe import EntsoePandasClient

from utils import RAW_DATA_DIR, ENTSOE_API_KEY

ENTSOE_PATH = os.path.join(RAW_DATA_DIR, "entsoe.csv")


def load_entsoe_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Load ENTSO-E raw data. If the data is not available, fetch it from the ENTSO-E API.
    :param start: Timestamp to start loading data from.
    :param end: Timestamp to stop loading data at.
    :return: DataFrame containing the ENTSO-E raw data.
    """
    if os.path.isfile(ENTSOE_PATH):
        logging.info(f"Loading ENTSO-E data from {ENTSOE_PATH}")
        df = pd.read_csv(ENTSOE_PATH, parse_dates=["timestamp"], index_col="timestamp")
    else:
        df = pd.DataFrame()

    full_range = pd.date_range(start, end, freq='15min', tz="UTC", inclusive="left")
    existing_range = df.index
    missing_range = full_range.difference(existing_range)

    if not missing_range.empty:
        missing_start = missing_range.min()
        missing_end = missing_range.max()
        country_codes = ["DE", "CZ"]
        new_df = __fetch_entsoe_data(missing_start, missing_end, country_codes)
        df = pd.concat([df, new_df])
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)

        logging.info(f"Saving ENTSO-E data to {ENTSOE_PATH}")
        df.to_csv(ENTSOE_PATH)

    return df.loc[full_range]


def __fetch_entsoe_data(start: pd.Timestamp, end: pd.Timestamp, country_codes: list[str]) -> pd.DataFrame:
    """
    Fetch ENTSO-E data for the time range for the given country codes.
    :param start: Timestamp to start fetching data from.
    :param end: Timestamp to stop fetching data at.
    :param country_codes: List of country codes to fetch data for.
    :return: DataFrame containing the fetched data.
    """
    logging.info(f"Fetching ENTSO-E data from {start} to {end} for {len(country_codes)} countries")

    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

    dataframes = []

    for country_code in country_codes:
        logging.info(f"Fetching ENTSO-E data for {country_code}")

        wind_and_solar_forecast = client.query_wind_and_solar_forecast(country_code, start=start, end=end)
        load_forecast = client.query_load_forecast(country_code, start=start, end=end)

        forecast_dataframe = pd.concat([wind_and_solar_forecast, load_forecast], axis=1)
        forecast_dataframe = forecast_dataframe.add_prefix(f"{country_code} ")
        forecast_dataframe.index.name = "timestamp"
        forecast_dataframe.index = forecast_dataframe.index.tz_convert("UTC")
        dataframes.append(forecast_dataframe)

    df = pd.concat(dataframes, axis=1)
    return df
