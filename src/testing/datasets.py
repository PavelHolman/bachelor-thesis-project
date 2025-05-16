import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

from src.features.load_day_ahead_dataset import load_day_ahead_dataset
from src.features.load_intra_day_dataset import load_intra_day_dataset
from src.testing.data_split import DA_KNOWN, ID_KNOWN, tabular_split, time_series_split, DA_TARGET, ID_TARGET

__start = pd.Timestamp("2024-07-01")
__end = pd.Timestamp("2025-04-01")
__val_cutoff = pd.Timestamp("2025-01-01")
__select_cutoff = pd.Timestamp("2025-02-01")
__test_cutoff = pd.Timestamp("2025-03-01")


def load_day_ahead_tabular_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the tabular day-ahead dataset and splits it into training, validation, select and test sets.
    Only the columns that are known for the day-ahead prediction (and the target) are kept.
    :return: Tuple containing the training, validation, select and test sets as DataFrames.
    """
    df = load_day_ahead_dataset(__start, __end)
    df = df[[col for col in df.columns if any(col.startswith(prefix) for prefix in DA_KNOWN)] + [DA_TARGET]]
    return tabular_split(df, __val_cutoff, __select_cutoff, __test_cutoff)


def load_day_ahead_time_series_split() -> tuple[
    TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """
    Loads the timeseries day-ahead dataset and splits it into training, validation, select and test sets.
    All the columns are kept, with not-known columns only being used as past data.
    :return: Tuple containing the training, validation, select and test sets as TimeSeriesDataFrames.
    """
    df = load_day_ahead_dataset(__start, __end)
    return time_series_split(df, __val_cutoff, __select_cutoff, __test_cutoff)


def load_intra_day_tabular_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the tabular intraday dataset and splits it into training, validation, select and test sets.
    Only the columns that are known for the intraday prediction (and the target) are kept.
    :return: Tuple containing the training, validation, select and test sets as DataFrames.
    """
    df = load_intra_day_dataset(__start, __end)
    df = df[[col for col in df.columns if any(col.startswith(prefix) for prefix in ID_KNOWN)] + [ID_TARGET]]
    return tabular_split(df, __val_cutoff, __select_cutoff, __test_cutoff)


def load_intra_day_time_series_split() -> tuple[
    TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """
    Loads the timeseries intraday dataset and splits it into training, validation, select and test sets.
    All the columns are kept, with not-known columns only being used as past data.
    :return: Tuple containing the training, validation, select and test sets as TimeSeriesDataFrames.
    """
    df = load_intra_day_dataset(__start, __end)
    return time_series_split(df, __val_cutoff, __select_cutoff, __test_cutoff)
