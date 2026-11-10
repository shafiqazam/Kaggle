import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from kagglekit import load_project, log_experiment, make_splits, make_submission, read_experiments, run_cv, score
from kagglekit import cli
from kagglekit.project import find_repo_root

REPO = find_repo_root()


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    shutil.copytree(REPO / "templates" / "project", tmp_path / "templates" / "project")
    return tmp_path


def make_project(workspace, slug="demo", **kwargs):
    root = cli.create_project(workspace, slug, **kwargs)
    return load_project(root / "notebooks")


def add_data(project, target_values, target="Transported"):
    project.data.mkdir(parents=True)
    pd.DataFrame({"id": [10, 11, 12], target: target_values}).to_csv(
        project.data / "sample_submission.csv", index=False
    )


def test_new_project_fills_template_and_is_found_from_subfolder(workspace):
    project = make_project(workspace, kind="competition", task="regression", metric="rmsle")
    assert (project.slug, project.kaggle_ref, project.task, project.metric) == (
        "demo", "demo", "regression", "rmsle",
    )
    leftovers = [
        path for path in project.root.rglob("*")
        if path.is_file() and "__" in path.read_text(encoding="utf-8").replace("__init__", "")
    ]
    assert leftovers == []


@pytest.mark.parametrize("slug", ["Bad Name", "../escape", ""])
def test_new_project_rejects_bad_slugs(workspace, slug):
    with pytest.raises(ValueError):
        cli.create_project(workspace, slug)


def test_new_project_will_not_overwrite(workspace):
    cli.create_project(workspace, "demo")
    with pytest.raises(FileExistsError):
        cli.create_project(workspace, "demo")


def test_time_splits_never_validate_on_the_past():
    for train_index, valid_index in make_splits(np.zeros(50), strategy="time"):
        assert train_index.max() < valid_index.min()


def test_stratified_splits_cover_every_row_once():
    y = np.array([0, 1] * 25)
    valid = np.concatenate([v for _, v in make_splits(y, n_splits=5)])
    assert sorted(valid) == list(range(50))


def test_run_cv_handles_boolean_targets_and_label_metrics():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y = (X["x"] > 0).to_numpy()
    result = run_cv(LogisticRegression, X, y, make_splits(y), "accuracy", "binary", X_test=X)
    assert result.mean > 0.9
    assert not np.isnan(result.oof).any()
    assert 0 <= result.test_pred.min() <= result.test_pred.max() <= 1


def test_rmsle_matches_definition():
    assert score("rmsle", [np.e - 1], [np.e**2 - 1]) == pytest.approx(1.0)


def test_submission_converts_probabilities_to_booleans(workspace):
    project = make_project(workspace, metric="accuracy")
    add_data(project, [False, False, True])
    path = make_submission(project, [0.9, 0.1, 0.5], "first try!")
    output = pd.read_csv(path)
    assert output["Transported"].tolist() == [True, False, True]
    assert path.parent == project.submissions
    assert "first-try" in path.name


def test_submission_keeps_probabilities_for_auc_and_reorders_by_id(workspace):
    project = make_project(workspace, metric="roc_auc")
    add_data(project, [0.5, 0.5, 0.5], target="y")
    output = pd.read_csv(make_submission(project, [0.3, 0.1, 0.2], "x", ids=[12, 10, 11]))
    assert output["y"].tolist() == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "predictions, ids",
    [([0.1, 0.2], None), ([0.1, np.nan, 0.2], None), ([0.1, 0.2, 0.3], [1, 2, 3])],
)
def test_submission_rejects_bad_input(workspace, predictions, ids):
    project = make_project(workspace)
    add_data(project, [0.5, 0.5, 0.5], target="y")
    with pytest.raises(ValueError):
        make_submission(project, predictions, "x", ids=ids)


def test_experiments_append(workspace):
    project = make_project(workspace)
    result = run_cv(
        LogisticRegression,
        pd.DataFrame({"x": range(40)}),
        np.array([0] * 20 + [1] * 20),
        make_splits(np.array([0] * 20 + [1] * 20)),
        "roc_auc",
        "binary",
    )
    log_experiment(project, "a", result, params={"C": 1}, notes="with, comma")
    log_experiment(project, "b", result)
    runs = read_experiments(project)
    assert runs["name"].tolist() == ["a", "b"]
    assert runs.loc[0, "notes"] == "with, comma"


def test_download_unzips_and_removes_archive(workspace, monkeypatch):
    project = make_project(workspace)
    calls = []

    def fake_kaggle(*args):
        calls.append(args)
        with zipfile.ZipFile(project.data / "demo.zip", "w") as handle:
            handle.writestr("train.csv", "a\n1\n")

    monkeypatch.setattr(cli, "_kaggle", fake_kaggle)
    cli.download(project)
    assert calls[0][:3] == ("competitions", "download", "-c")
    assert (project.data / "train.csv").is_file()
    assert not list(project.data.glob("*.zip"))


def test_submit_needs_confirmation(workspace, monkeypatch, tmp_path):
    project = make_project(workspace)
    file = tmp_path / "sub.csv"
    file.write_text("id,y\n")
    calls = []
    monkeypatch.setattr(cli, "_kaggle", lambda *args: calls.append(args))
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert cli.submit(project, file, "msg") is False
    assert calls == []
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert cli.submit(project, file, "msg") is True
    assert calls[0][:2] == ("competitions", "submit")


def test_every_existing_project_config_loads():
    from kagglekit.project import all_projects

    projects = all_projects(REPO)
    assert {p.slug for p in projects} >= {"titanic", "spaceship-titanic", "hourly-energy-consumption"}
    assert all(p.root.is_dir() for p in projects)
