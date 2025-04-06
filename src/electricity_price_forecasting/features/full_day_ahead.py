import os

import pandas as pd

from src.electricity_price_forecasting.features.generate_day_ahead_features import generate_day_ahead_lag_features, \
    generate_day_ahead_rolling_features
from src.electricity_price_forecasting.features.generate_time_features import generate_time_features
from utils import PROCESSED_DATA_DIR


def full_day_ahead_features() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "quarterly.csv"), index_col=0, parse_dates=True)
    df.drop(columns=[col for col in df.columns if "ID" in col], inplace=True)
    generate_day_ahead_lag_features(df)
    generate_day_ahead_rolling_features(df)
    generate_time_features(df)
    df = df.copy()
    return df
