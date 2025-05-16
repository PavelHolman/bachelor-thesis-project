import numpy as np
from autogluon.core.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from autogluon.core.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_eval_metrics():
    """
    Gets all the same evaluation metrics as in the thesis.
    :return: List of evaluation metrics.
    """
    return [
        mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, directional_accuracy,
        peak_detection_accuracy, peak_detection_precision, peak_detection_recall, peak_detection_f1
    ]


def __directional_accuracy(y_true: np.array, y_pred: np.array):
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction)


def __peak_detection_accuracy(y_true: np.array, y_pred: np.array, q: float = 0.9):
    threshold = np.quantile(y_true, q)
    true_peaks = (y_true > threshold).astype(int)
    pred_peaks = (y_pred > threshold).astype(int)
    return accuracy_score(true_peaks, pred_peaks)


def __peak_detection_precision(y_true: np.array, y_pred: np.array, q: float = 0.9):
    threshold = np.quantile(y_true, q)
    true_peaks = (y_true > threshold).astype(int)
    pred_peaks = (y_pred > threshold).astype(int)
    return precision_score(true_peaks, pred_peaks)


def __peak_detection_recall(y_true: np.array, y_pred: np.array, q: float = 0.9):
    threshold = np.quantile(y_true, q)
    true_peaks = (y_true > threshold).astype(int)
    pred_peaks = (y_pred > threshold).astype(int)
    return recall_score(true_peaks, pred_peaks)


def __peak_detection_f1(y_true: np.array, y_pred: np.array, q: float = 0.9):
    threshold = np.quantile(y_true, q)
    true_peaks = (y_true > threshold).astype(int)
    pred_peaks = (y_pred > threshold).astype(int)
    return f1_score(true_peaks, pred_peaks)


directional_accuracy = make_scorer(
    score_func=__directional_accuracy,
    greater_is_better=True,
    name="directional accuracy",
)

peak_detection_accuracy = make_scorer(
    score_func=__peak_detection_accuracy,
    greater_is_better=True,
    name="peak detection accuracy",
)

peak_detection_precision = make_scorer(
    score_func=__peak_detection_precision,
    greater_is_better=True,
    name="peak detection precision",
)

peak_detection_recall = make_scorer(
    score_func=__peak_detection_recall,
    greater_is_better=True,
    name="peak detection recall",
)

peak_detection_f1 = make_scorer(
    score_func=__peak_detection_f1,
    greater_is_better=True,
    name="peak detection f1",
)
