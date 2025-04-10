import holidays
import pandas as pd


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-based features for the given DataFrame.
    :param df: DataFrame containing the data.
    :return: DataFrame with generated time-based features.
    """
    time_df = pd.DataFrame(index=df.index)

    cz_holidays = holidays.country_holidays("CZ", years=df.index.year.unique()).keys()

    time_df["Hour of day"] = df.index.hour
    time_df["Day of week"] = df.index.dayofweek
    time_df["Month"] = df.index.month
    time_df["Year"] = df.index.year
    time_df['Is holiday'] = [(x in cz_holidays) for x in df.index.date]

    time_df["Day of week"] = time_df["Day of week"].astype("category")
    time_df['Is holiday'] = time_df['Is holiday'].astype("category")

    return time_df
