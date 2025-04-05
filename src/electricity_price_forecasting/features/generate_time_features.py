import holidays
import pandas as pd


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year

    cz_holidays = holidays.country_holidays("CZ", years=df.index.year.unique())
    df['is_holiday'] = df.index.date.isin(cz_holidays)

    return df
