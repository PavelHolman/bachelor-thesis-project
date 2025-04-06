import pandas as pd


def generate_lag_day_ahead_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in range(1, 7 * 24 + 1):
        df[f"lag_{lag}"] = df["DA Price (EUR/MWh)"].shift(lag)

    return df


def generate_lag_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in range(1, 24 * 4 + 1):
        df[f"lag_{lag}"] = df["ID Weighted average of the prices (EUR/MWh)"].shift(lag)

    return df


def generate_rolling_day_ahead_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in range(24, 24 * 7 + 1, 24):
        df[f"rolling_mean_{window}"] = df["DA Price (EUR/MWh)"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["DA Price (EUR/MWh)"].rolling(window=window).std()

    return df


def generate_rolling_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in range(24 * 4, 24 * 4 * 7 + 1, 24 * 4):
        df[f"rolling_mean_{window}"] = df["ID Weighted average of the prices (EUR/MWh)"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["ID Weighted average of the prices (EUR/MWh)"].rolling(window=window).std()

    return df
