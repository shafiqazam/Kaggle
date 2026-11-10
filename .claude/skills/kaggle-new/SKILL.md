---
name: kaggle-new
description: Start a new Kaggle project in this repo. Use when the user wants to begin a competition, dataset analysis, or course project.
---

Start a project by following these steps from the repo root.

1. Get the Kaggle slug from the user's message or URL. For example, `https://www.kaggle.com/competitions/titanic` has the slug `titanic`. Ask only if you cannot tell the slug, or if the task or metric is unclear. The competition's Evaluation page gives the metric.
2. Run `uv run kk new <slug> --task <binary|regression> --metric <metric>`. Add `--kind dataset --ref owner/name` for datasets, or `--kind learning` for courses.
3. From the new project folder, run `uv run kk download`. The user may need to accept the competition rules on Kaggle first. If the download fails on authentication or rules, report that and stop.
4. Fill in the Problem section of the project's `README.md`.
5. Run the EDA notebook, then state a validation plan before any modeling: which split strategy matches how test differs from train, and why.

Do not submit anything to Kaggle as part of this skill.
