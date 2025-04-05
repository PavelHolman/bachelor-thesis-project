from typing import Tuple

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


def train_val_test_split(
        df: pd.DataFrame,
        val: pd.Timestamp = pd.Timestamp("2025-01-01"),
        test: pd.Timestamp = pd.Timestamp("2025-03-01"),
) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    df = df.copy()
    df = df.sort_index()
    df["item_id"] = "intraday"
    df["timestamp"] = df.index.tz_convert(None)

    train = df[df["timestamp"] < val]
    val = df[(df["timestamp"] >= val) & (df["timestamp"] < test)]
    test = df[df["timestamp"] >= test]

    train = TimeSeriesDataFrame.from_data_frame(train)
    val = TimeSeriesDataFrame.from_data_frame(val)
    test = TimeSeriesDataFrame.from_data_frame(test)
    return train, val, test
