import pandas as pd


def generate_intraday_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    day = 24 * 4
    for lag in range(1, 7 * day + 1):
        df[f"da_lag_{lag}"] = df["DA Price (EUR/MWh)"].shift(lag)
        df[f"id_lag_{lag}"] = df["ID Weighted average of the prices (EUR/MWh)"].shift(lag)

    return df


def generate_intraday_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    day = 24 * 4
    for window in range(day, 7 * day + 1, day):
        df[f"da_rolling_mean_{window}"] = df["DA Price (EUR/MWh)"].rolling(window=window).mean()
        df[f"da_rolling_std_{window}"] = df["DA Price (EUR/MWh)"].rolling(window=window).std()

        df[f"id_rolling_mean_{window}"] = df["ID Weighted average of the prices (EUR/MWh)"].rolling(window=window).mean()
        df[f"id_rolling_std_{window}"] = df["ID Weighted average of the prices (EUR/MWh)"].rolling(window=window).std()

    return df
