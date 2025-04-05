from typing import Tuple

import pandas as pd


def train_val_test_split(
        df: pd.DataFrame,
        val: pd.Timestamp = pd.Timestamp("2025-01-01"),
        test: pd.Timestamp = pd.Timestamp("2025-03-01"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df = df.sort_index()

    train_df = df[df["timestamp"] < val]
    val_df = df[(df["timestamp"] >= val) & (df["timestamp"] < test)]
    test_df = df[df["timestamp"] >= test]

    return train_df, val_df, test_df
