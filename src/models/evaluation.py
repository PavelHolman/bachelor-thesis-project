import logging

import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.splitter import ExpandingWindowSplitter

__eval_days = 14


def predict_timeseries_model(predictor: TimeSeriesPredictor, data: TimeSeriesDataFrame) -> dict[str, pd.DataFrame]:
    prediction_length = predictor.prediction_length
    freq = predictor.freq if predictor.freq[0].isdigit() else "1" + predictor.freq
    intervals_per_day = int(pd.Timedelta(days=1) // pd.Timedelta(freq))
    n_days = prediction_length // intervals_per_day
    predictions = {}
    for model_name in predictor.model_names():
        logging.info(f"Predicting with {model_name}...")
        model_predictions = pd.DataFrame()
        splitter = ExpandingWindowSplitter(
            prediction_length=prediction_length,
            num_val_windows=__eval_days,
            val_step_size=intervals_per_day
        )
        for (past_data, future_data) in splitter.split(data):
            weekly_prediction = predictor.predict(data=past_data, known_covariates=future_data, model=model_name)
            daily_predictions = [weekly_prediction.slice_by_timestep(d * intervals_per_day, (d + 1) * intervals_per_day)
                                 for d in range(n_days)]
            daily_means = pd.DataFrame(
                {f"{predictor.target} (T+{i})": d["mean"].values for (i, d) in enumerate(daily_predictions)},
                index=daily_predictions[0].index.get_level_values(1))
            model_predictions = pd.concat([model_predictions, daily_means], axis=0)
        predictions[model_name] = model_predictions
    return predictions


def predict_tabular_model(predictor: TabularPredictor, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred_data = data[data.index > (data.index[-1] - pd.Timedelta(days=__eval_days))]
    predictions = {}
    for model_name in predictor.model_names():
        logging.info(f"Predicting with {model_name}...")
        predictions[model_name] = predictor.predict(data=pred_data, model=model_name)
    return predictions
