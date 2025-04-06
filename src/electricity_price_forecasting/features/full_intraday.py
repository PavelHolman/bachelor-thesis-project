import os

import pandas as pd

from src.electricity_price_forecasting.features.generate_intraday_features import generate_intraday_lag_features, \
    generate_intraday_rolling_features
from src.electricity_price_forecasting.features.generate_time_features import generate_time_features
from utils import PROCESSED_DATA_DIR


def full_intraday() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "merged.csv"), index_col=0, parse_dates=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    generate_intraday_lag_features(df)
    generate_intraday_rolling_features(df)
    generate_time_features(df)
    df = df.copy()
    return df
