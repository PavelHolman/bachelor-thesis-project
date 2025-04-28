import logging
import pandas as pd
from autogluon.core.metrics import Scorer
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer
from autogluon.timeseries.splitter import ExpandingWindowSplitter
import seaborn as sns
from matplotlib import pyplot as plt


def evaluate_timeseries_model(
        predictor: TimeSeriesPredictor,
        train: TimeSeriesDataFrame,
        val: TimeSeriesDataFrame,
        select: TimeSeriesDataFrame,
        metrics: list[TimeSeriesScorer],
):
    target = predictor.target
    prediction_length = predictor.prediction_length

    model_names = predictor.model_names()

    for name in model_names:
        fig, axes = plt.subplots(2, 3, figsize=(32, 18))
        for ((i, data_name), data) in zip(enumerate(["Training", "Validation", "Selection"]), [train, val, select]):
            logging.info(f"Evaluating {name} on {data_name} data...")
            preds = pd.DataFrame()
            num_val_windows = 3
            splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=num_val_windows)
            for (past_data, future_data) in splitter.split(data):
                preds = pd.concat([preds, predictor.predict(data=past_data, known_covariates=future_data, model=name)])

            y_index = [ts for (_, ts) in data.index[-num_val_windows * prediction_length:].values]
            y_true = data[target][-num_val_windows * prediction_length:].values
            y_pred = preds["mean"].values

            __plot_predictions(y_index, y_true, y_pred, target, name, data_name, axes[0, i], axes[1, i])
        plt.plot()


def evaluate_tabular_model(
        predictor: TabularPredictor,
        train: pd.DataFrame,
        val: pd.DataFrame,
        select: pd.DataFrame,
        metrics: list[Scorer],
):
    target = predictor.label

    model_names = predictor.model_names()

    for name in model_names:
        fig, axes = plt.subplots(2, 3, figsize=(32, 18))
        for ((i, data_name), data) in zip(enumerate(["Training", "Validation", "Selection"]), [train, val, select]):
            logging.info(f"Evaluating {name} on {data_name} data...")

            y_index = [ts for (_, ts) in data.index.values]
            y_true = data[target].values
            y_pred = predictor.predict(data=data, model=name)

            __plot_predictions(y_index, y_true, y_pred, target, name, data_name, axes[0, i], axes[1, i])
        plt.plot()


def __plot_predictions(
        y_index: list,
        y_true: pd.Series,
        y_pred: pd.Series,
        target: str,
        name: str,
        data_name: str,
        ax1: plt.Axes,
        ax2: plt.Axes,
):
    sns.lineplot(x=y_index, y=y_true, ax=ax1)
    sns.lineplot(x=y_index, y=y_pred, ax=ax1)
    ax1.set_title(f"'{name}' Predictions ({data_name})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(target)
    ax1.legend(["True", None, "Predicted"])

    errors = y_pred - y_true
    sns.kdeplot(errors, ax=ax2)
    ax2.set_title(f"'{name}' Error Distribution ({data_name})")
    ax2.set_xlabel("Error")
    ax2.set_ylabel("Density")
    ax2.legend(["Error"])
    ax2.axvline(0, color="red", linestyle="--")
    ax2.grid()
