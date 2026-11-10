"""The `kk` command: create projects, download data, submit, and list projects."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from kagglekit.cv import METRICS
from kagglekit.experiments import read_experiments
from kagglekit.project import KIND_FOLDERS, Project, all_projects, find_repo_root, load_project

TEXT_SUFFIXES = {".md", ".toml", ".ipynb"}


def create_project(
    repo_root: Path,
    slug: str,
    kind: str = "competition",
    ref: str | None = None,
    task: str = "binary",
    metric: str | None = None,
) -> Path:
    """Copy the template to a new project folder and fill in its settings."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("Use a lowercase slug with letters, numbers and hyphens.")
    if kind == "dataset" and not ref:
        raise ValueError("Datasets need --ref owner/dataset-name.")
    metric = metric or ("roc_auc" if task == "binary" else "rmse")
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {sorted(METRICS)}.")

    destination = repo_root / KIND_FOLDERS[kind] / slug
    if destination.exists():
        raise FileExistsError(f"{destination} already exists.")

    replacements = {
        "__SLUG__": slug,
        "__KIND__": kind,
        "__REF__": ref if ref is not None else (slug if kind == "competition" else ""),
        "__TASK__": task,
        "__METRIC__": metric,
    }
    shutil.copytree(repo_root / "templates" / "project", destination)
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            for placeholder, value in replacements.items():
                text = text.replace(placeholder, value)
            path.write_text(text, encoding="utf-8", newline="\n")
    return destination


def _kaggle(*args: str) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("The kaggle command was not found. Run this with `uv run kk`.")
    subprocess.run([executable, *args], check=True)


def download(project: Project) -> None:
    """Download a project's data and unzip it."""
    if not project.kaggle_ref:
        raise ValueError(f"{project.slug} has no kaggle_ref in project.toml.")
    project.data.mkdir(parents=True, exist_ok=True)
    if project.kind == "competition":
        _kaggle("competitions", "download", "-c", project.kaggle_ref, "-p", str(project.data))
    elif project.kind == "dataset":
        _kaggle("datasets", "download", "-d", project.kaggle_ref, "-p", str(project.data))
    else:
        raise ValueError(f"{project.kind} projects have nothing to download.")
    for archive in project.data.glob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(project.data)
        archive.unlink()


def submit(project: Project, file: Path, message: str, assume_yes: bool = False) -> bool:
    """Upload a submission after confirmation, since submissions are limited per day."""
    if project.kind != "competition":
        raise ValueError("Only competition projects can submit.")
    if not file.is_file():
        raise FileNotFoundError(file)
    if not assume_yes:
        answer = input(f"Submit {file.name} to {project.kaggle_ref}? [y/N] ")
        if answer.strip().lower() != "y":
            print("Cancelled.")
            return False
    _kaggle("competitions", "submit", "-c", project.kaggle_ref, "-f", str(file), "-m", message)
    return True


def scores(project: Project) -> None:
    """Show this competition's submissions and leaderboard scores from Kaggle."""
    if project.kind != "competition":
        raise ValueError("Only competition projects have submissions.")
    _kaggle("competitions", "submissions", "-c", project.kaggle_ref)


def list_projects() -> None:
    rows = []
    for project in all_projects():
        runs = read_experiments(project)
        best = ""
        if len(runs) and project.metric in METRICS:
            scores = runs["cv_mean"]
            value = scores.max() if METRICS[project.metric].maximize else scores.min()
            best = f"{value:.5f}"
        rows.append((project.kind, project.slug, project.metric or "-", str(len(runs)), best or "-"))
    headers = ("KIND", "PROJECT", "METRIC", "RUNS", "BEST CV")
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kk", description="Kaggle workspace tools")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="create a project from the template")
    new.add_argument("slug", help="project folder name; for competitions, the Kaggle slug")
    new.add_argument("--kind", choices=sorted(KIND_FOLDERS), default="competition")
    new.add_argument("--ref", help="Kaggle reference, required for datasets (owner/name)")
    new.add_argument("--task", choices=["binary", "regression"], default="binary")
    new.add_argument("--metric", choices=sorted(METRICS))

    commands.add_parser("download", help="download data for the current project")

    send = commands.add_parser("submit", help="submit a file for the current project")
    send.add_argument("file", type=Path)
    send.add_argument("-m", "--message", required=True)
    send.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    commands.add_parser("scores", help="show Kaggle submissions for the current project")
    commands.add_parser("list", help="list projects and their best CV scores")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "new":
            path = create_project(
                find_repo_root(), args.slug, args.kind, args.ref, args.task, args.metric
            )
            print(f"Created {path}")
        elif args.command == "download":
            download(load_project())
        elif args.command == "submit":
            submit(load_project(), args.file, args.message, args.yes)
        elif args.command == "scores":
            scores(load_project())
        elif args.command == "list":
            list_projects()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"kk: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
