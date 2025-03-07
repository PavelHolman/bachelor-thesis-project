import os
import logging
import pandas as pd
from functools import reduce

from electricity_price_forecasting.data.fetch_entsoe import fetch_load_forecast, fetch_generation_forecast, \
    fetch_wind_and_solar_forecast
from electricity_price_forecasting.data.fetch_ote_cr import fetch_day_ahead_data, fetch_intra_day_data
from electricity_price_forecasting.data.fetch_weather import fetch_weather_data
from utils import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_data(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if start >= end:
        logging.error("Start date must be before end date")
        raise ValueError("Start date must be before end date")

    logging.info("Fetching 'CZ' data from ENTSO-E...")
    load_forecast_df = fetch_load_forecast("CZ", start, end)
    generation_forecast_df = fetch_generation_forecast("CZ", start, end)
    wind_solar_forecast_df = fetch_wind_and_solar_forecast("CZ", start, end)

    logging.info("Fetching data from OTE CR...")
    day_ahead_df = fetch_day_ahead_data(start, end)
    intra_day_df = fetch_intra_day_data(start, end)

    logging.info("Fetching weather data...")
    weather_df = fetch_weather_data(start, end)

    logging.info("Preprocessing data...")
    merged_df = merge_data(
        load_forecast_df, generation_forecast_df, wind_solar_forecast_df, day_ahead_df, intra_day_df, weather_df
    )

    merged_df.index = pd.to_datetime(merged_df.index, utc=True)
    merged_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "merged.csv"))
    logging.info("Data saved to 'merged.csv'")

    quarterly_df = quarterly_aggregate(merged_df)
    quarterly_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "quarterly.csv"))
    logging.info("Quarterly data saved to 'quarterly.csv'")

    hourly_df = hourly_aggregate(merged_df)
    hourly_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "hourly.csv"))
    logging.info("Hourly data saved to 'hourly.csv'")

    return merged_df, quarterly_df, hourly_df


def merge_data(*dataframes: pd.DataFrame) -> pd.DataFrame:
    return reduce(lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True), dataframes)


def quarterly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    hourly_cols = [col for col in df.columns if df[col][df.index.minute != 0].nunique() == 0]
    df_hourly = df[df.index.minute == 0][hourly_cols]
    quarterly_cols = [col for col in df.columns if col not in hourly_cols]

    df_quarterly = df[quarterly_cols].groupby(df.index.floor("h")).apply(
        lambda x: pd.DataFrame(x.values.reshape(4, -1).T,
                               columns=[0, 15, 30, 45],
                               index=x.columns)
    ).unstack(level=1)
    df_quarterly.columns = [f"{col} ({m})" for m, col in df_quarterly.columns]

    df = df_hourly.merge(df_quarterly, left_index=True, right_index=True)

    return df.reindex(sorted(df.columns), axis=1)


def hourly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df_hourly = df.resample("h").mean()
    return df_hourly.reindex(sorted(df_hourly.columns), axis=1)


if __name__ == "__main__":
    fetch_data(pd.Timestamp("2024-07-01"), pd.Timestamp("2025-03-06"))
