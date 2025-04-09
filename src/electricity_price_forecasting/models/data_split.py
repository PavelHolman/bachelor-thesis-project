from typing import Tuple

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


def train_val_test_split(
        df: pd.DataFrame,
        val_cutoff: pd.Timestamp,
        select_cutoff: pd.Timestamp,
        test_cutoff: pd.Timestamp,
) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    df = df.copy()
    df = df.sort_index()
    df["item_id"] = "intraday"
    df["timestamp"] = df.index.tz_convert(None)

    train = df[df["timestamp"] < val_cutoff]
    val = df[(df["timestamp"] >= val_cutoff) & (df["timestamp"] < select_cutoff)]
    select = df[(df["timestamp"] >= select_cutoff) & (df["timestamp"] < test_cutoff)]
    test = df[df["timestamp"] >= test_cutoff]

    train = TimeSeriesDataFrame.from_data_frame(train)
    val = TimeSeriesDataFrame.from_data_frame(val)
    select = TimeSeriesDataFrame.from_data_frame(select)
    test = TimeSeriesDataFrame.from_data_frame(test)
    return train, val, select, test
