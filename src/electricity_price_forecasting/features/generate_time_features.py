import holidays
import pandas as pd


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_week"] = df["day_of_week"].astype("category")
    df["month"] = df.index.month
    df["year"] = df.index.year

    cz_holidays = holidays.country_holidays("CZ", years=df.index.year.unique()).keys()
    df['is_holiday'] = [(x in cz_holidays) for x in df.index.date]
    df['is_holiday'] = df['is_holiday'].astype("category")

    return df
