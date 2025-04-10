import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


def shuffled_split(
        df: pd.DataFrame,
        val_size: float,
        test_size: float,
        eval_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into shuffled training, validation, test and evaluation sets.
    :param df: The input DataFrame containing the time series data.
    :param val_size: The proportion of the data to include in the validation set.
    :param test_size: The proportion of the data to include in the test set.
    :param eval_size: The proportion of the data to include in the evaluation set.
    :return: A tuple containing the training, validation, test, and evaluation sets as DataFrames.
    """
    train = df.sample(frac=1 - (val_size + test_size + eval_size))
    remaining = df.drop(train.index)
    val = remaining.sample(frac=val_size / (val_size + test_size + eval_size))
    remaining = remaining.drop(val.index)
    test = remaining.sample(frac=test_size / (test_size + eval_size))
    evaluation = remaining.drop(test.index)

    return train, val, test, evaluation


def time_series_split(
        df: pd.DataFrame,
        val_cutoff: pd.Timestamp,
        test_cutoff: pd.Timestamp,
        eval_cutoff: pd.Timestamp,
) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """
    Splits the DataFrame into continuous training, validation, test and evaluation sets for time series forecasting.
    :param df: The input DataFrame containing the time series data.
    :param val_cutoff: The cutoff timestamp for the validation set.
    :param test_cutoff: The cutoff timestamp for the test set.
    :param eval_cutoff: The cutoff timestamp for the evaluation set.
    :return: A tuple containing the training, validation, test, and evaluation sets as TimeSeriesDataFrames.
    """
    df["item_id"] = 1
    df["timestamp"] = df.index.tz_convert(None)

    train = TimeSeriesDataFrame.from_data_frame(df[df["timestamp"] < val_cutoff])
    val = TimeSeriesDataFrame.from_data_frame(df[(df["timestamp"] >= val_cutoff) & (df["timestamp"] < test_cutoff)])
    test = TimeSeriesDataFrame.from_data_frame(df[(df["timestamp"] >= test_cutoff) & (df["timestamp"] < eval_cutoff)])
    evaluation = TimeSeriesDataFrame.from_data_frame(df[df["timestamp"] >= eval_cutoff])

    return train, val, test, evaluation
