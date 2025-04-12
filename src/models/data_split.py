import logging

import pandas as pd
from autogluon.common import TabularDataset
from autogluon.timeseries import TimeSeriesDataFrame

DA_KNOWN = ["T", "WF"]
ID_KNOWN = ["T", "WF", "DA", "PG"]


def tabular_split(
        df: pd.DataFrame,
        val_cutoff: pd.Timestamp,
        test_cutoff: pd.Timestamp,
        eval_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, test and evaluation sets for tabular data.
    :param df: The input DataFrame containing the tabular data.
    :param val_cutoff: The cutoff timestamp for the validation set.
    :param test_cutoff: The cutoff timestamp for the test set.
    :param eval_cutoff: The cutoff timestamp for the evaluation set.
    :return: A tuple containing the training, validation, test, and evaluation sets as TabularDatasets.
    """
    df["item_id"] = 1
    timestamp = df.index.tz_convert(None)

    train = TabularDataset(df[timestamp < val_cutoff])
    val = TabularDataset(df[(timestamp >= val_cutoff) & (timestamp < test_cutoff)])
    test = TabularDataset(df[(timestamp >= test_cutoff) & (timestamp < eval_cutoff)])
    evaluation = TabularDataset(df[timestamp >= eval_cutoff])

    __log_data_split_info(df, train, val, test, evaluation)
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

    __log_data_split_info(df, train, val, test, evaluation)
    return train, val, test, evaluation


def __log_data_split_info(
        df: pd.DataFrame,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        evaluation: pd.DataFrame
):
    logging.info(
        f"Data split info:\n"
        f"Original DataFrame size: {len(df)}\n"
        f"Train size: {len(train)} ({len(train) / len(df):.2%})\n"
        f"Validation size: {len(val)} ({len(val) / len(df):.2%})\n"
        f"Test size: {len(test)} ({len(test) / len(df):.2%})\n"
        f"Evaluation size: {len(evaluation)} ({len(evaluation) / len(df):.2%})"
    )
