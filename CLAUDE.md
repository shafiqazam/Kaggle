# Kaggle workspace

This repo is developed from multiple devices, with GitHub as the source of truth.

## Rules

- NEVER commit a `.env` file or `kaggle.json`. Only templates with placeholder values, such as `.env.example`, may be committed.
- Manage Python only through `uv`. Add packages with `uv add <package>`, run code with `uv run <command>`, and do not use `pip install` or edit `uv.lock` by hand.
- Do not change the Python version in `.python-version` or the pins in `pyproject.toml` unless asked.

## Syncing between devices

- At the start of a session, run `git pull --ff-only`, then `uv sync`. If the working tree is dirty or the pull fails, stop and tell the user before changing anything.
- Before leaving a device, commit and push. Notebooks do not merge cleanly, so avoid having unpushed notebook changes on two devices.
- Machine-only state stays off Git: `.venv/`, `.env`, `kaggle.json`, and `.claude/settings.local.json`.

## New device setup

1. Install `git` and `uv`. On macOS, also run `brew install libomp`, which XGBoost needs.
2. Clone the repo and run `uv sync`. Add `--group agents` for the AI Agents course.
3. Copy secrets over manually, never through GitHub: `AI Agents Intensive Course Google/.env` (see its `.env.example`) and `~/.kaggle/kaggle.json`.
4. Select the repo's `.venv` as the notebook kernel.
