import logging
import os

import pandas as pd
from autogluon.core.metrics import Scorer
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.splitter import ExpandingWindowSplitter

from utils import RESULTS_DIR

__eval_days = 14


def predict_timeseries_model(
        predictor: TimeSeriesPredictor, data: TimeSeriesDataFrame, cache: str
) -> dict[str, pd.DataFrame]:
    prediction_length = predictor.prediction_length
    freq = predictor.freq if predictor.freq[0].isdigit() else "1" + predictor.freq
    intervals_per_day = int(pd.Timedelta(days=1) // pd.Timedelta(freq))
    n_days = prediction_length // intervals_per_day
    predictions = {}
    cache_path = os.path.join(RESULTS_DIR, cache)
    os.makedirs(cache_path, exist_ok=True)

    for model_name in predictor.model_names():
        result_path = os.path.join(cache_path, model_name)
        if os.path.isfile(result_path):
            logging.info(f"Loading cached predictions for {model_name}")
            predictions[model_name] = pd.read_csv(result_path, index_col=0, parse_dates=True)
            continue

        logging.info(f"Predicting with {model_name}...")
        model_predictions = pd.DataFrame()
        splitter = ExpandingWindowSplitter(
            prediction_length=prediction_length,
            num_val_windows=__eval_days,
            val_step_size=intervals_per_day
        )
        for (past_data, future_data) in splitter.split(data):
            full_prediction = predictor.predict(data=past_data, known_covariates=future_data, model=model_name)
            part_predictions = [full_prediction.slice_by_timestep(d * intervals_per_day, (d + 1) * intervals_per_day)
                                for d in range(n_days)]
            part_means = pd.DataFrame(
                {f"{predictor.target} (T+{i})": d["mean"].values for (i, d) in enumerate(part_predictions)},
                index=part_predictions[0].index.get_level_values(1))
            model_predictions = pd.concat([model_predictions, part_means], axis=0)
        predictions[model_name] = model_predictions
        model_predictions.to_csv(result_path)
    return predictions


def predict_tabular_model(predictor: TabularPredictor, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred_data = data[data.index > (data.index[-1] - pd.Timedelta(days=__eval_days))]
    predictions = {}
    for model_name in predictor.model_names():
        logging.info(f"Predicting with {model_name}...")
        predictions[model_name] = predictor.predict(data=pred_data, model=model_name)
    return predictions


def evaluate(
        predictions: dict[str, pd.DataFrame],
        true_values: pd.Series,
        metrics: list[Scorer],
) -> pd.DataFrame:
    results = {}
    for model_name, pred in predictions.items():
        results[model_name] = {}

        true_like_predictions = pd.DataFrame(index=pred.index)
        for i, col in enumerate(pred.columns):
            true_like_predictions[col] = true_values[pred.index + pd.Timedelta(days=i)].values

        for metric in metrics:
            results[model_name][f"{metric.name}"] = metric(
                y_true=true_like_predictions.stack(),
                y_pred=pred.stack()
            )

            for i, col in enumerate(pred.columns):
                results[model_name][f"{metric.name} (T+{i})"] = metric(
                    y_true=true_like_predictions[col],
                    y_pred=pred[col]
                )
    return pd.DataFrame(results).T
