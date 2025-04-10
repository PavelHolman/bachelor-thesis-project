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
    train_data = df.sample(frac=1 - (val_size + test_size + eval_size))
    remaining_data = df.drop(train_data.index)
    val_data = remaining_data.sample(frac=val_size / (val_size + test_size + eval_size))
    remaining_data = remaining_data.drop(val_data.index)
    test_data = remaining_data.sample(frac=test_size / (test_size + eval_size))
    eval_data = remaining_data.drop(test_data.index)

    return train_data, val_data, test_data, eval_data


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
    train_data = TimeSeriesDataFrame.from_data_frame(df[df.index < val_cutoff])
    val_data = TimeSeriesDataFrame.from_data_frame(df[(df.index >= val_cutoff) & (df.index < test_cutoff)])
    test_data = TimeSeriesDataFrame.from_data_frame(df[(df.index >= test_cutoff) & (df.index < eval_cutoff)])
    eval_data = TimeSeriesDataFrame.from_data_frame(df[df.index >= eval_cutoff])

    return train_data, val_data, test_data, eval_data
