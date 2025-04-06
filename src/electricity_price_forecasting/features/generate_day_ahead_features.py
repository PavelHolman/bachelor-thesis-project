import pandas as pd


def generate_day_ahead_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    day = 24
    for lag in range(1, 7 * day + 1):
        df[f"da_lag_{lag}"] = df["DA Price (EUR/MWh)"].shift(lag)

    return df



def generate_day_ahead_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    day = 24
    for window in range(day, 7 * day + 1, day):
        df[f"da_rolling_mean_{window}"] = df["DA Price (EUR/MWh)"].rolling(window=window).mean()
        df[f"da_rolling_std_{window}"] = df["DA Price (EUR/MWh)"].rolling(window=window).std()

    return df
