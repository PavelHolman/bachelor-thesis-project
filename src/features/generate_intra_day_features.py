import pandas as pd


def generate_intra_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lag and rolling intraday features for the given DataFrame.
    :param df: DataFrame containing the data.
    :return: DataFrame with generated intraday features.
    """
    intra_day_df = pd.DataFrame(index=df.index)

    intra_day_series = df["ID Weighted average of the prices (EUR/MWh)"]
    intervals_per_day = int(pd.Timedelta("1D") / pd.Timedelta(df.index.inferred_freq))

    for lag in range(1, 7 * intervals_per_day + 1):
        intra_day_df[f"ID Lag {lag}"] = intra_day_series.shift(lag)

    for window in range(intervals_per_day, 7 * intervals_per_day + 1, intervals_per_day):
        intra_day_df[f"ID Rolling mean {window}"] = intra_day_series.rolling(window=window).mean()
        intra_day_df[f"ID Rolling std {window}"] = intra_day_series.rolling(window=window).std()

    return intra_day_df
