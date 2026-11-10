# Kaggle workspace

This repo is the home for all Kaggle work. It is developed from multiple devices, with GitHub as the source of truth.

## Layout

- `competitions/<slug>/`, `datasets/<slug>/`, `learning/<slug>/`: one folder per project, each with a `project.toml`.
- `src/kagglekit/`: shared code, importable from any notebook. `kk` is its command line.
- `templates/project/`: what `kk new` copies.
- Older projects keep data beside their notebooks, with `data_dir = "."` in `project.toml`. Do not reorganize them unless asked.

## Project workflow

1. Start with `/kaggle-new`, or `uv run kk new <slug>` then `uv run kk download` from the project folder.
2. Explore in `notebooks/01_eda.ipynb`, then write the validation plan in the project's `README.md` before modeling.
3. Get a baseline score from `notebooks/02_baseline.ipynb`.
4. Iterate one idea at a time, in numbered notebooks. Compare on CV using the same splits, and log every run with `log_experiment`.
5. Build files with `make_submission`, and submit with `/kaggle-submit`. Never submit without the user's confirmation.
6. When finished, fill in the README's final result. Move code to `kagglekit` only once a second project needs it, and add a test.

Check all project status with `uv run kk list`, and run `uv run pytest` after changing `kagglekit`.

## Rules

- NEVER commit a `.env` file, `kaggle.json`, or `access_token`. Only templates with placeholder values, such as `.env.example`, may be committed.
- This repo is public. Keep data for new projects in the ignored `data/` folder. Do not commit competition data.
- Fit preprocessing inside each CV fold, for example in a sklearn Pipeline passed to `run_cv`.
- Manage Python only through `uv`. Add packages with `uv add <package>`, run code with `uv run <command>`, and do not use `pip install` or edit `uv.lock` by hand.
- Do not change the Python version in `.python-version` or the pins in `pyproject.toml` unless asked.

## Syncing between devices

- At the start of a session, run `git pull --ff-only`, then `uv sync`. If the working tree is dirty or the pull fails, stop and tell the user before changing anything.
- Before leaving a device, commit and push. Notebooks do not merge cleanly, so avoid having unpushed notebook changes on two devices.
- Commit each project's `experiments.csv`, because it is the shared record of runs.
- Machine-only state stays off Git: `.venv/`, `.env`, Kaggle credentials, `data/`, `submissions/`, `models/`, and `.claude/settings.local.json`.

## New device setup

1. Install `git` and `uv`. On macOS, also run `brew install libomp`, which XGBoost needs.
2. Clone the repo and run `uv sync`. Add `--group agents` for the AI Agents course.
3. Copy secrets over manually, never through GitHub: `learning/ai-agents-intensive/.env` (see its `.env.example`) and your Kaggle token. Save a new-style `KGAT_` token as `~/.kaggle/access_token`, containing only the token.
4. Select the repo's `.venv` as the notebook kernel.

### Windows differences

Run the same `uv` and `git` commands in PowerShell or Git Bash. Only these steps change:

- Install the tools with `winget install Git.Git astral-sh.uv Microsoft.VCRedist.2015+.x64`. Git for Windows is required by Claude Code, and XGBoost needs the Visual C++ runtime. `libomp` is not needed.
- Before cloning, run `git config --global core.longpaths true`, and clone to a short path such as `C:\dev\Kaggle`. Packages in the `agents` group have deep folder paths that can exceed the Windows path limit.
- The notebook kernel is `.venv\Scripts\python.exe`, and the Kaggle token goes in `%USERPROFILE%\.kaggle\access_token`.
- On Windows on ARM, the locked packages have no native builds for Python 3.14. Use x64 Python instead: `uv sync --python cpython-3.14-windows-x86_64-none`.
- If using WSL, follow the Linux path: clone inside the WSL filesystem, not under `/mnt/c`, and keep it separate from any Windows clone because a `.venv` cannot be shared between them.
