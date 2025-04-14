import holidays
import pandas as pd


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-based features for the given DataFrame such as hour of day,
    day of week, month, year, and whether the date is a holiday.
    :param df: DataFrame containing the data.
    :return: DataFrame with generated time-based features.
    """
    time_df = pd.DataFrame(index=df.index)

    cz_holidays = holidays.country_holidays("CZ", years=df.index.year.unique()).keys()

    time_df["T Hour of day"] = df.index.hour
    time_df["T Day of week"] = df.index.dayofweek
    time_df["T Month"] = df.index.month
    time_df["T Year"] = df.index.year
    time_df["T Is holiday"] = [(x in cz_holidays) for x in df.index.date]

    time_df["T Day of week"] = time_df["T Day of week"].astype("category")
    time_df["T Is holiday"] = time_df["T Is holiday"].astype("category")

    return time_df
