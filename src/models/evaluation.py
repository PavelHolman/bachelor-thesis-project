import logging

import numpy as np
import pandas as pd
from autogluon.core.metrics import Scorer
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer
from autogluon.timeseries.splitter import ExpandingWindowSplitter
import seaborn as sns
from matplotlib import pyplot as plt

__eval_days = 14


def evaluate_timeseries_model(
        predictor: TimeSeriesPredictor,
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        select: TimeSeriesDataFrame,
        metrics: list[Scorer],
) -> pd.DataFrame:
    target = predictor.target
    prediction_length = predictor.prediction_length
    model_names = predictor.model_names()
    freq = predictor.freq if predictor.freq[0].isdigit() else "1" + predictor.freq
    intervals_per_day = int(pd.Timedelta(days=1) // pd.Timedelta(freq))
    n_days = prediction_length // intervals_per_day
    splits = ["Training", "Validation", "Selection"]
    metrics_df = pd.DataFrame(
        columns=["model"] + [f"{metric.name} [{s} ({d})]" for metric in metrics for s in splits for d in range(n_days)])

    for model_name in model_names:
        fig, axes = plt.subplots(n_days * 2, 3, figsize=(32, n_days * 9))
        metric_results = {}
        for ((i, split), data) in zip(enumerate(splits), [train, val, select]):
            logging.info(f"Evaluating {model_name} on {split} data...")
            preds = [pd.DataFrame() for _ in range(n_days)]
            splitter = ExpandingWindowSplitter(
                prediction_length=prediction_length,
                num_val_windows=__eval_days,
                val_step_size=intervals_per_day
            )
            for (past_data, future_data) in splitter.split(data):
                predictions = predictor.predict(data=past_data, known_covariates=future_data, model=model_name)
                for d in range(n_days):
                    preds[d] = pd.concat([preds[d], predictions[intervals_per_day * d:intervals_per_day * (d + 1)]])

            for d in range(n_days):
                last_timestamp = data.index.values[-1][1]
                past_one_day = last_timestamp.floor("D") + pd.Timedelta(days=1)
                last_day = past_one_day - pd.Timedelta(days=(n_days - d - 1))
                first_day = last_day - pd.Timedelta(days=__eval_days)
                date_mask = (data.index.get_level_values(1) >= first_day) & (data.index.get_level_values(1) < last_day)
                range_data = data[date_mask]
                y_index = range_data.index.get_level_values(1)
                y_true = range_data[target]
                y_pred = preds[d]["mean"]

                __plot_pred(y_index, y_true, y_pred, target, model_name, split, d, axes[2 * d, i], axes[2 * d + 1, i])
                for metric in metrics:
                    metric_results[(metric, split, d)] = metric(y_true, y_pred)

        metrics_df.loc[len(metrics_df)] = [model_name] + [metric_results[(m, s, d)]
                                                          for m in metrics for s in splits for d in range(n_days)]
    plt.plot()
    return metrics_df


def evaluate_tabular_model(
        predictor: TabularPredictor,
        train: pd.DataFrame,
        val: pd.DataFrame,
        select: pd.DataFrame,
        metrics: list[Scorer],
) -> pd.DataFrame:
    splits = ["Training", "Validation", "Selection"]
    metrics_df = pd.DataFrame(columns=["model"] + [f"{metric.name} [{s}]" for metric in metrics for s in splits])
    target = predictor.label
    model_names = predictor.model_names()

    for model_name in model_names:
        fig, axes = plt.subplots(2, 3, figsize=(32, 18))
        metric_results = {}
        for ((i, split), data) in zip(enumerate(splits), [train, val, select]):
            data = data[data.index > (data.index[-1] - pd.Timedelta(days=__eval_days))]
            logging.info(f"Evaluating {model_name} on {split} data...")

            y_index = data.index.values
            y_true = data[target].values
            y_pred = predictor.predict(data=data, model=model_name)

            __plot_pred(y_index, y_true, y_pred, target, model_name, split, 1, axes[0, i], axes[1, i])
            for metric in metrics:
                metric_results[(metric, split)] = metric(y_true, y_pred)

        metrics_df.loc[len(metrics_df)] = [model_name] + [metric_results[(m, s)] for m in metrics for s in splits]
    plt.plot()
    return metrics_df


def __plot_pred(
        y_index: pd.Index,
        y_true: pd.Series,
        y_pred: pd.Series,
        target: str,
        model_name: str,
        split: str,
        day: int,
        ax1: plt.Axes,
        ax2: plt.Axes,
):
    sns.lineplot(x=y_index, y=y_true, ax=ax1)
    sns.lineplot(x=y_index, y=y_pred, ax=ax1)
    ax1.set_title(f"[{day}] '{model_name}' Predictions ({split})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(target)
    ax1.legend(["True", None, "Predicted"])

    errors = y_pred - y_true
    sns.kdeplot(errors, ax=ax2)
    ax2.set_title(f"[{day}] '{model_name}' Error Distribution ({split})")
    ax2.set_xlabel("Error")
    ax2.set_ylabel("Density")
    ax2.legend(["Error"])
    ax2.axvline(0, color="red", linestyle="--")
    ax2.grid()
