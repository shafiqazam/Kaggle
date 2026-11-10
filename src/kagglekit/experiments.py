"""Keep one experiments.csv per project, committed so every device shares it."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from kagglekit.cv import CVResult
from kagglekit.project import Project

COLUMNS = [
    "timestamp",
    "name",
    "metric",
    "cv_mean",
    "cv_std",
    "folds",
    "git_commit",
    "params",
    "submission",
    "public_lb",
    "notes",
]


def _git_commit(cwd: Path) -> str:
    """Return the short commit, marked when there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return f"{commit}+dirty" if dirty else commit


def log_experiment(
    project: Project,
    name: str,
    result: CVResult,
    params: dict | None = None,
    submission: Path | None = None,
    notes: str = "",
) -> None:
    """Append one run to the project's experiments.csv."""
    is_new = not project.experiments_path.exists()
    with open(project.experiments_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "name": name,
                "metric": result.metric,
                "cv_mean": f"{result.mean:.6f}",
                "cv_std": f"{result.std:.6f}",
                "folds": len(result.fold_scores),
                "git_commit": _git_commit(project.root),
                "params": json.dumps(params or result.params, sort_keys=True, default=str),
                "submission": submission.name if submission else "",
                "public_lb": "",
                "notes": notes,
            }
        )


def read_experiments(project: Project) -> pd.DataFrame:
    """Return the project's runs, or an empty table when there are none."""
    if not project.experiments_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(project.experiments_path)
