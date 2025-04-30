import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import MAE, MAPE, RMSE
from autogluon.core.metrics import mean_absolute_error as mae, root_mean_squared_error as rmse, \
    mean_absolute_percentage_error as mape
from autogluon.core.metrics import make_scorer
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer


def get_tabular_metrics():
    return [mae, mape, rmse, tab_weighted_mae, tab_directional_accuracy, tab_peak_detection, tab_cost_simulation]


def get_timeseries_metrics():
    return [MAE(), MAPE(), RMSE(), TSWeightedMAE(), TSDirectionalAccuracy(), TSPeakDetection(), TSCostSimulation()]


def __weighted_mae(y_true, y_pred):
    weights = np.abs(y_true) / np.mean(np.abs(y_true))
    return np.mean(weights * np.abs(y_true - y_pred))


def __directional_accuracy(y_true, y_pred):
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction)


def __peak_detection(y_true, y_pred, threshold=0.9):
    true_peaks = y_true > np.quantile(y_true, threshold)
    pred_peaks = y_pred > np.quantile(y_pred, threshold)
    return np.mean(true_peaks == pred_peaks)


def __cost_simulation(y_true, y_pred, base_load=100.0):
    true_cost = np.sum(y_true * base_load)
    pred_cost = np.sum(y_pred * base_load)
    return np.abs(true_cost - pred_cost) / true_cost


tab_weighted_mae = make_scorer(
    score_func=__weighted_mae,
    greater_is_better=False,
    name="weighted_mae",
)
tab_directional_accuracy = make_scorer(
    score_func=__directional_accuracy,
    greater_is_better=True,
    name="directional_accuracy",
)
tab_peak_detection = make_scorer(
    score_func=__peak_detection,
    greater_is_better=True,
    name="peak_detection",
)
tab_cost_simulation = make_scorer(
    score_func=__cost_simulation,
    greater_is_better=False,
    name="cost_simulation",
)


class TSWeightedMAE(TimeSeriesScorer):
    name = "weighted_mae"
    greater_is_better = False

    def compute_metric(
            self,
            data_future: TimeSeriesDataFrame,
            predictions: TimeSeriesDataFrame,
            target: str = "target",
            **kwargs,
    ) -> float:
        y_true = data_future[target].values
        y_pred = predictions["mean"].values
        return __weighted_mae(y_true, y_pred)


class TSDirectionalAccuracy(TimeSeriesScorer):
    name = "directional_accuracy"
    greater_is_better = True

    def compute_metric(
            self,
            data_future: TimeSeriesDataFrame,
            predictions: TimeSeriesDataFrame,
            target: str = "target",
            **kwargs,
    ) -> float:
        y_true = data_future[target].values
        y_pred = predictions["mean"].values
        return __directional_accuracy(y_true, y_pred)


class TSPeakDetection(TimeSeriesScorer):
    name = "peak_detection"
    greater_is_better = True

    def compute_metric(
            self,
            data_future: TimeSeriesDataFrame,
            predictions: TimeSeriesDataFrame,
            target: str = "target",
            **kwargs,
    ) -> float:
        y_true = data_future[target].values
        y_pred = predictions["mean"].values
        return __peak_detection(y_true, y_pred)


class TSCostSimulation(TimeSeriesScorer):
    name = "cost_simulation"
    greater_is_better = False

    def compute_metric(
            self,
            data_future: TimeSeriesDataFrame,
            predictions: TimeSeriesDataFrame,
            target: str = "target",
            **kwargs,
    ) -> float:
        y_true = data_future[target].values
        y_pred = predictions["mean"].values
        return __cost_simulation(y_true, y_pred)
