import numpy as np
from autogluon.core.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from autogluon.core.metrics import make_scorer


def get_eval_metrics():
    return [mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error,
            tab_weighted_mae, tab_directional_accuracy, tab_peak_detection, tab_cost_simulation]


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
