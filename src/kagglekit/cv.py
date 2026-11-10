"""Cross-validation that gives every model the same splits and scoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn import metrics as skm
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


@dataclass(frozen=True)
class Metric:
    fn: Callable
    needs_labels: bool = False
    maximize: bool = True


METRICS = {
    "accuracy": Metric(skm.accuracy_score, needs_labels=True),
    "f1": Metric(skm.f1_score, needs_labels=True),
    "roc_auc": Metric(skm.roc_auc_score),
    "log_loss": Metric(skm.log_loss, maximize=False),
    "mae": Metric(skm.mean_absolute_error, maximize=False),
    "rmse": Metric(skm.root_mean_squared_error, maximize=False),
    "rmsle": Metric(skm.root_mean_squared_log_error, maximize=False),
}


def score(metric: str, y_true, y_pred, threshold: float = 0.5) -> float:
    """Score predictions, converting probabilities to labels when the metric needs it."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {sorted(METRICS)}.")
    spec = METRICS[metric]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if spec.needs_labels:
        y_true = y_true.astype(int)
        y_pred = (y_pred >= threshold).astype(int)
    return float(spec.fn(y_true, y_pred))


def make_splits(y, strategy: str = "stratified", n_splits: int = 5, groups=None, seed: int = 42):
    """Return a list of (train_index, validation_index) pairs.

    Use "time" for data sorted by time, so validation always comes after training.
    """
    y = np.asarray(y)
    placeholder = np.zeros(len(y))
    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(placeholder, y))
    if strategy == "kfold":
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(placeholder))
    if strategy == "group":
        if groups is None:
            raise ValueError("The group strategy needs `groups`.")
        return list(GroupKFold(n_splits=n_splits).split(placeholder, y, groups))
    if strategy == "time":
        return list(TimeSeriesSplit(n_splits=n_splits).split(placeholder))
    raise ValueError(f"Unknown strategy {strategy!r}.")


@dataclass
class CVResult:
    metric: str
    oof: np.ndarray
    fold_scores: list[float]
    test_pred: np.ndarray | None = None
    params: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores))

    def __repr__(self) -> str:
        return f"CVResult({self.metric}={self.mean:.5f} +/- {self.std:.5f}, folds={len(self.fold_scores)})"


def _predict(model, X, task: str) -> np.ndarray:
    if task == "binary":
        return model.predict_proba(X)[:, 1]
    if task == "regression":
        return np.asarray(model.predict(X), dtype=float)
    raise ValueError(f"run_cv supports binary and regression tasks, not {task!r}.")


def run_cv(
    make_model: Callable[[], object],
    X,
    y,
    splits,
    metric: str,
    task: str,
    X_test=None,
) -> CVResult:
    """Fit a new model on each split and collect out-of-fold and test predictions.

    `make_model` returns a fresh unfitted model. Put fitted preprocessing inside it,
    such as in a sklearn Pipeline, so nothing leaks from the validation fold.
    """
    y = np.asarray(y)
    if y.dtype == bool:
        y = y.astype(int)
    oof = np.full(len(y), np.nan)
    fold_scores = []
    test_folds = []
    for train_index, valid_index in splits:
        model = make_model()
        model.fit(X.iloc[train_index], y[train_index])
        oof[valid_index] = _predict(model, X.iloc[valid_index], task)
        fold_scores.append(score(metric, y[valid_index], oof[valid_index]))
        if X_test is not None:
            test_folds.append(_predict(model, X_test, task))
    test_pred = np.mean(test_folds, axis=0) if test_folds else None
    return CVResult(metric=metric, oof=oof, fold_scores=fold_scores, test_pred=test_pred)
