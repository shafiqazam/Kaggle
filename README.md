# Kaggle workspace

All my Kaggle competitions, dataset projects and course work, with one shared environment and toolkit.

```text
competitions/<slug>/   Kaggle competitions
datasets/<slug>/       Dataset projects
learning/<slug>/       Courses and tutorials
src/kagglekit/         Shared code and the kk command
templates/project/     Starting point for new projects
```

## Start a project

```bash
uv sync
uv run kk new spaceship-titanic --task binary --metric accuracy
cd competitions/spaceship-titanic
uv run kk download
```

Each new project has this shape:

```text
project.toml        Kaggle reference, task and metric
README.md           Problem, validation plan and findings
experiments.csv     Log of every run, created on the first run
notebooks/          01_eda and 02_baseline to start
data/               Downloaded data, not in Git
submissions/        Generated files, not in Git
models/             Saved models, not in Git
```

## Use the toolkit in a notebook

```python
from kagglekit import load_project, log_experiment, make_splits, make_submission, run_cv

project = load_project()
train = project.read_train()
splits = make_splits(train[project.target_column()], strategy="stratified")
```

`uv run kk list` shows every project and its best CV score. See `CLAUDE.md` for the full workflow and device setup.
