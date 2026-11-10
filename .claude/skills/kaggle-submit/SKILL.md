---
name: kaggle-submit
description: Submit a prediction file to a Kaggle competition from this repo and record the result. Use when the user asks to submit to Kaggle.
---

Daily submissions are limited and each one is permanent on Kaggle, so follow these steps in order.

1. Find the file in the project's `submissions/` folder and its row in `experiments.csv`. Refuse a file with no logged CV score unless the user explicitly accepts that.
2. Check that no fitted preprocessing in the notebook used validation or test data.
3. Show the user the file name, CV score, and the best previous CV and leaderboard scores. Ask for confirmation and wait for a clear yes.
4. From the project folder, run `uv run kk submit submissions/<file> -m "<experiment name>, CV <score>" --yes`.
5. From the project folder, run `uv run kk scores` to read the public score, then fill in `public_lb` on that row of `experiments.csv`.
6. Report the CV and leaderboard scores together. If the gap grew compared with earlier runs, say that the validation plan may need another look.
