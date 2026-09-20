# 🏏 Cricket Performance Analysis & Prediction System

Web application for the **Python for Data Science — Team Mini Project**.
This is the *application development* half of the project: it takes the
Random Forest models your teammate trained and wraps them in a
Streamlit UI so a user can get predictions two ways:

1. **Manual entry** — fill in a single player's stats, or pick a team.
2. **CSV upload (batch)** — upload a file with many rows and get
   predictions for all of them at once, with a downloadable results file.

## Folder structure

```
cricket_app/
├── app.py                     # the Streamlit application
├── requirements.txt
├── README.md
├── models/                    # your teammate's trained artifacts
│   ├── random_forest_performance_model.pkl
│   ├── role_encoder.pkl
│   ├── team_encoder.pkl
│   ├── feature_columns.pkl
│   ├── team_random_forest_performance_model.pkl
│   └── team_feature_columns.pkl
└── dataset/
    └── ODI_Cricket_Data_new.csv
```

Keep this structure — `app.py` loads models from `models/` and the
reference dataset from `dataset/` using relative paths.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`.

## Features

- **Player — Manual Entry**: form for batting/bowling/match stats →
  predicted class (Good / Average / Poor) with probability breakdown
  and a confidence chart.
- **Team — Select & Predict**: pick any team from the dataset, see its
  aggregated stats, predict its category, and rank *all* teams by
  predicted "Good" probability.
- **Batch Upload (CSV)**:
  - Download a ready-made template, fill in as many player rows as you
    like, upload it, and get predictions + probabilities for every row
    (with a summary chart and a CSV download of the results).
  - Rows with a `role`/`team` value the encoders never saw during
    training are skipped and shown separately rather than crashing
    the app.
  - There's also a second uploader for *raw match-level data* (same
    shape as the original dataset) — it aggregates it per team and
    runs the team-level model on the result.
- **Dataset Explorer**: quick EDA charts (top run scorers, runs vs.
  strike rate, team comparisons) so this can double as part of your
  EDA section too.

## Notes for the report / viva

- Both models are `RandomForestClassifier` (scikit-learn), predicting
  one of three classes: `Good`, `Average`, `Poor`.
- Player-level features: `role, total_runs, strike_rate,
  total_balls_faced, total_wickets_taken, total_runs_conceded,
  total_overs_bowled, total_matches_played, matches_played_as_batter,
  matches_played_as_bowler, matches_won, matches_lost,
  player_of_match_awards, average, team`.
- Team-level features are the same list minus `role` and `team`,
  aggregated (sum for counts, mean for rates) across all players of
  that team.
- `role_encoder.pkl` / `team_encoder.pkl` are `LabelEncoder`s fit on
  the training data — any unseen category in a batch upload is
  flagged rather than silently mis-encoded.
