"""Find a project from any folder inside it and read its project.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONFIG_NAME = "project.toml"
KIND_FOLDERS = {"competition": "competitions", "dataset": "datasets", "learning": "learning"}


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repo root by walking up from `start`."""
    current = (start or Path(__file__)).resolve()
    for folder in [current, *current.parents]:
        if (folder / "pyproject.toml").is_file() and (folder / "templates" / "project").is_dir():
            return folder
    raise FileNotFoundError(f"No Kaggle workspace found above {current}")


@dataclass(frozen=True)
class Project:
    root: Path
    slug: str
    kind: str
    kaggle_ref: str = ""
    task: str = ""
    metric: str = ""
    target: str = ""
    id_column: str = ""
    data_dir: str = "data"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_submission_file: str = "sample_submission.csv"

    @property
    def data(self) -> Path:
        return (self.root / self.data_dir).resolve()

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def experiments_path(self) -> Path:
        return self.root / "experiments.csv"

    def read_train(self, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.data / self.train_file, **kwargs)

    def read_test(self, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.data / self.test_file, **kwargs)

    def read_sample_submission(self, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.data / self.sample_submission_file, **kwargs)

    def target_column(self) -> str:
        """Use the configured target, or the second sample submission column."""
        if self.target:
            return self.target
        return str(pd.read_csv(self.data / self.sample_submission_file, nrows=0).columns[1])

    def id_name(self) -> str:
        """Use the configured ID, or the first sample submission column."""
        if self.id_column:
            return self.id_column
        return str(pd.read_csv(self.data / self.sample_submission_file, nrows=0).columns[0])


def read_project(root: Path) -> Project:
    """Build a Project from the project.toml in `root`."""
    with open(root / CONFIG_NAME, "rb") as handle:
        config = tomllib.load(handle)
    allowed = set(Project.__dataclass_fields__) - {"root"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {root / CONFIG_NAME}: {sorted(unknown)}")
    return Project(root=root, **config)


def load_project(start: Path | None = None) -> Project:
    """Return the project containing `start`, which defaults to the current folder."""
    current = (start or Path.cwd()).resolve()
    for folder in [current, *current.parents]:
        if (folder / CONFIG_NAME).is_file():
            return read_project(folder)
    raise FileNotFoundError(f"No {CONFIG_NAME} found above {current}. Run this inside a project.")


def all_projects(repo_root: Path | None = None) -> list[Project]:
    """Return every project in the repo, sorted by folder."""
    root = repo_root or find_repo_root()
    projects = []
    for folder in KIND_FOLDERS.values():
        for config in sorted((root / folder).glob(f"*/{CONFIG_NAME}")):
            projects.append(read_project(config.parent))
    return projects
