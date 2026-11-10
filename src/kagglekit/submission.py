"""Build submission files that match the competition's sample submission."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from kagglekit.cv import METRICS
from kagglekit.project import Project


def make_submission(
    project: Project,
    predictions,
    name: str,
    ids=None,
    threshold: float = 0.5,
) -> Path:
    """Validate predictions against the sample submission and write a CSV.

    Pass `ids` when predictions may not be in the sample submission's row order.
    """
    sample = project.read_sample_submission()
    id_name, target = sample.columns[0], sample.columns[1]
    values = np.asarray(predictions)

    if len(values) != len(sample):
        raise ValueError(f"Got {len(values)} predictions, but the sample has {len(sample)} rows.")
    if np.issubdtype(values.dtype, np.number) and np.isnan(values.astype(float)).any():
        raise ValueError("Predictions contain NaN.")

    if ids is not None:
        ids = np.asarray(ids)
        order = {value: index for index, value in enumerate(ids)}
        if len(order) != len(ids) or set(order) != set(sample[id_name]):
            raise ValueError("`ids` must match the sample submission's IDs exactly.")
        values = values[[order[value] for value in sample[id_name]]]

    metric = METRICS.get(project.metric)
    if project.task == "binary" and metric is not None and metric.needs_labels:
        labels = values >= threshold
        values = labels if sample[target].dtype == bool else labels.astype(int)

    output = sample[[id_name, target]].copy()
    output[target] = values

    project.submissions.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "submission"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = project.submissions / f"{stamp}_{safe_name}.csv"
    output.to_csv(path, index=False, lineterminator="\n")
    return path
