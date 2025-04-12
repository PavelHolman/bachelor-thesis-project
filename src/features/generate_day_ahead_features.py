import pandas as pd


def generate_day_ahead_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lag and rolling day-ahead features for the given DataFrame.
    :param df: DataFrame containing the data.
    :return: DataFrame with generated day-ahead features.
    """
    day_ahead_df = pd.DataFrame(index=df.index)

    day_ahead_series = df["DA Price (EUR/MWh)"]
    freq = df.index.inferred_freq if df.index.inferred_freq[0].isdigit() else f"1{df.index.inferred_freq}"
    intervals_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))

    for lag in range(1, 7 * intervals_per_day + 1):
        day_ahead_df[f"DA Lag {lag}"] = day_ahead_series.shift(lag)

    for window in range(intervals_per_day, 7 * intervals_per_day + 1, intervals_per_day):
        day_ahead_df[f"DA Rolling mean {window}"] = day_ahead_series.rolling(window=window).mean()
        day_ahead_df[f"DA Rolling std {window}"] = day_ahead_series.rolling(window=window).std()

    return day_ahead_df
