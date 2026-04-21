# Football Match Predictor — Code Explainer Report

This document explains the code you are presenting, file-by-file, in practical language.
It focuses on the core files we built/extended for your project and the `predictions_*.json` output format.

---

## 1. High-level architecture

The project runs in this order:

1. **Collect raw match data** from ESPN API (`src/collect_data.py`)
2. **Store data** in MySQL (`teams`, `matches`)
3. **Generate engineered features** (`src/generate_features.py` -> `match_features`)
4. **Train models** (`src/train_models.py`)
5. **Predict upcoming matches + store predictions** (`src/predict_upcoming.py`)
6. **Visualize outputs and accuracy** (`dashboard/app.py`)

---

## 2. File-by-file explanation

## 2.1 `src/feature_columns.py`

### What it does
Defines one central list of all training features:
`TRAINING_FEATURE_COLUMNS`.

### Why it matters
This avoids inconsistencies across:
- feature generation
- training
- prediction
- validation scripts

If you add/remove a feature, all downstream code reads from this same list.

---

## 2.2 `src/collect_data.py`

### Purpose
Collect fixtures/results from ESPN and keep the DB clean and up to date.

### Key constants
- `_ESPN_BASE`: endpoint base URL
- `_VALID_RESULTS`: allowed labels (`H`, `D`, `A`)
- `_MAX_REASONABLE_GOALS`: quality rule
- `_STALE_UPCOMING_GRACE_HOURS`: avoids keeping unresolved fixtures that are already in the past

### Main utility functions
- `_to_utc_naive(...)`: normalizes datetimes for DB comparisons
- `_is_stale_upcoming(...)`: detects unresolved fixtures that are already in the past
- `normalize_team_name(...)`: reduces name duplicates from whitespace variations
- `_safe_int(...)`: safe parse for API score fields
- `_derive_result(...)`: maps score to H/D/A
- `_parse_match_date(...)`: robust ESPN date parsing

### Team resolution
- `_get_or_create_team(...)`: fetches team ID or inserts new team row

### API event cleaning
- `clean_match(event)`:
  - extracts home/away teams and date
  - sets goals/result only for finished matches
  - rejects stale unresolved fixtures
  - returns normalized row dict for DB insert

### Validation and insert
- `validate_match(match)` checks:
  - valid IDs, valid score ranges
  - score/result consistency
  - no stale unresolved fixture

- `insert_matches(matches)`:
  - **upserts** rows into `matches`
  - if fixture already exists, updates goals/result when they become available

### Historical + upcoming fetch
- `fetch_historical_data(...)`: monthly range pulls
- `fetch_upcoming_data(...)`: lookahead fixture pulls

### Stale cleanup
- `remove_stale_upcoming_matches(...)`:
  - deletes unresolved past fixtures
  - also deletes linked `match_features` + `match_predictions` rows

### Entry point
- `main()` parses CLI args and runs:
  1. stale cleanup
  2. historical fetch
  3. upcoming fetch

---

## 2.3 `src/generate_features.py`

### Purpose
Builds predictive features using only information available **before** each match (no leakage).

### Time normalization
- `_to_naive(...)`: keeps datetime handling consistent in DB queries

### Core feature builders
- `calculate_team_form(team_id, before_date, num_matches)`:
  - recent points, goals, goal difference

- `calculate_head_to_head(home_id, away_id, before_date)`:
  - historical H2H counts

- `calculate_rest_fatigue(team_id, before_date)`:
  - days since last match
  - matches in last 7 / 14 days
  - last venue (home/away)

- `calculate_venue_strength(team_id, before_date, venue, num_matches=10)`:
  - venue-specific PPG + goal diff average

- `calculate_overall_strength(team_id, before_date, num_matches=30)`:
  - longer-term overall PPG + goal diff

- `calculate_elo_before_match(home_id, away_id, before_date)`:
  - rolling Elo rating snapshot before match
  - caches by timestamp for efficiency

- `calculate_travel_penalty(rest, is_away_team)`:
  - heuristic fatigue/travel pressure signal

### Master assembler
- `generate_features_for_match(...)`:
  - combines all feature blocks into one row dict
  - includes base form/H2H + fatigue + home/away strength + Elo

### Schema protection
- `ensure_match_features_columns()`:
  - checks DB columns
  - auto-adds missing feature columns via `ALTER TABLE`

### Bulk generation
- `generate_all_features()`:
  - reads all matches chronologically
  - computes features row-by-row
  - upserts into `match_features`

---

## 2.4 `src/train_models.py`

### Purpose
Trains and compares two models on historical labeled matches.

### Data loading
- `load_training_data()`:
  - joins `match_features` with `matches.result`
  - only finished matches (`result IS NOT NULL`)

### Training flow
- `train_and_evaluate_models()`:
  1. load `X`, `y`
  2. split **80/20** with stratification
  3. train Random Forest
  4. train Logistic Regression (after scaling)
  5. evaluate (accuracy, precision, recall, F1)
  6. save models and metadata

### Model configs
- Random Forest:
  - `n_estimators=100`, `max_depth=10`, etc.
- Logistic Regression:
  - multinomial, `lbfgs`, `max_iter=1000`

### Outputs
- `models/random_forest_model.pkl`
- `models/logistic_regression_model.pkl`
- `models/feature_scaler.pkl`
- `models/models_metadata.json`
- `models/feature_importance.json`

### Current metadata (latest)
- train samples: **697**
- test samples: **175**
- RF accuracy: **54.86%**
- LR accuracy: **52.57%**
- best model: **Random Forest**

---

## 2.5 `src/validate_form_windows.py`

### Purpose
Validates different form windows (N recent matches) with repeated random seeds.

### Main functions
- `load_finished_matches()`: base rows for validation
- `build_dataset_for_window(...)`: rebuilds features for selected `form_window`
- `evaluate_window_for_seed(...)`: trains/evaluates RF+LR for one seed
- `summarize_window_results(...)`: mean/std/min/max aggregation
- `print_results_table(...)`: presentation-friendly summary
- `save_results(...)`: writes JSON results

### Why it matters
Prevents “lucky split” conclusions; gives stable estimates.

### Current 100-seed summary (windows 4,6,7,9,15,20)
- Best RF mean test: **window 7**
- Best LR mean test: **window 6**

Saved in:
- `models/form_window_validation_repeated.json`

---

## 2.6 `src/player_stats.py`

### Purpose
Fetches roster/player stats and injury/unavailability info from ESPN.

### Data flow
- `fetch_league_teams(...)` -> find league teams
- `resolve_team_id(...)` -> map team name to ESPN team ID
- `fetch_team_roster(...)` -> pull roster payload
- `roster_to_dataframes(...)` -> build:
  - `players` DataFrame
  - `unavailable` DataFrame

### Availability impact logic
- `compute_availability_impact(players_df, unavailable_df)`:
  - injured count
  - top scorer absence count
  - missing goal share
  - combined availability penalty

- `get_team_availability_impact(...)`:
  - safe wrapper with fallback if API errors

### CLI utility
- `main()` prints top player stats and injury report.

---

## 2.7 `src/predict_upcoming.py`

### Purpose
Predict upcoming matches with both models, estimate scoreline, store predictions, and reconcile accuracy.

### Model loading
- `load_models()` reads RF + LR + scaler from disk.

### Scoreline helpers
- `_get_goal_baseline()` uses historical average goals
- `estimate_scoreline(...)` computes expected score with availability penalties

### Outcome prediction helpers
- `_primary_prediction(predictions)` chooses consensus or highest-confidence model

### Persistence helpers
- `create_prediction_run(...)` inserts run metadata
- `store_prediction(...)` inserts/upserts per-match prediction
- `reconcile_resolved_predictions()`:
  - finds pending predictions where match result now exists
  - fills actuals + correctness fields
  - marks status `resolved`
- `get_accuracy_summary()` returns aggregate accuracy stats

### Feature consistency
- `build_feature_input(...)` aligns match features with `TRAINING_FEATURE_COLUMNS`

### Main prediction function
- `predict_upcoming_matches(league_slug, form_window)`:
  1. reconcile old pending predictions
  2. query upcoming matches
  3. generate features + RF/LR predictions
  4. compute scoreline + player availability impact
  5. store everything in DB + JSON output file

### Important filter
Upcoming query only includes:
- `m.result IS NULL`
- `m.match_date >= UTC_TIMESTAMP()`

This prevents past unresolved fixtures from being predicted.

---

## 2.8 `dashboard/app.py`

### Purpose
Presentation-friendly Streamlit interface with separate pages.

### Pages
- `🎯 Dashboard`: headline system stats
- `🔮 Predictions`: upcoming matches, predicted score, RF/LR outcomes + confidence
- `✅ Accuracy`: separate evaluation page (model metrics/charts/history)
- `📈 Statistics`: team summary table
- `👤 Player Stats`: roster/injury explorer
- `🏥 Data Quality`: quality counters

### Accuracy page (`✅ Accuracy`)
Shows:
- resolved match count
- outcome accuracy
- exact score accuracy
- RF/LR/Consensus bar chart
- monthly trend line
- recent resolved prediction table with correctness icons

This is your main “proof of model quality” page for class.

---

## 2.9 SQL schema files

### `sql/schema.sql` and `sql/features_schema.sql`

Define/create:
- base tables (`teams`, `matches`)
- ML feature table (`match_features`)
- prediction tracking (`prediction_runs`, `match_predictions`)

Also include uniqueness/quality constraints (e.g., fixture uniqueness).

---

## 2.10 `predictions/predictions_*.json` (output format)

Generated by `save_predictions(...)` in `predict_upcoming.py`.

Top-level keys:
- `timestamp`
- `total_predictions`
- `predictions` (list)

Per-match structure:
- `match_id`, `date`, `home_team`, `away_team`
- `predictions`:
  - `random_forest.prediction` + `confidence`
  - `logistic_regression.prediction` + `confidence`
  - `consensus.prediction` + `agreement`
- `predicted_score.home_goals`, `predicted_score.away_goals`
- `player_availability.home/away` block:
  - `injured_players`
  - `top_scorer_absences`
  - `missing_goal_share`
  - `availability_penalty`
  - `key_absences`
  - `available`

Example interpretation:
- If RF says `A` with 0.49 and LR says `A` with 0.85,
  both favor away win, LR is more certain for that outcome class.

---

## 3. “What is used where?” (quick slide version)

## Training uses
- historical engineered numeric features from `match_features` (28 columns)
- target = final match result (`H/D/A`)

## Prediction uses
- same model feature set
- plus live availability impact for scoreline adjustment

## Dashboard shows
- upcoming predictions
- model confidence and disagreement/agreement
- separate accuracy/evaluation page

---

## 4. Current known caveats (mention in presentation)

1. Historical injury snapshots are limited in ESPN feed, so injury data is stronger for **live prediction context** than historical supervised training.
2. Logistic Regression may emit numeric runtime warnings during repeated validation runs (model still trains/evaluates).
3. Accuracy accumulation needs true pre-match stored predictions before results arrive.

---

## 5. Suggested 3-minute class walkthrough

1. Explain pipeline (collect -> features -> train -> predict -> evaluate)
2. Show feature groups (form, fatigue, strength, Elo)
3. Show model comparison (RF vs LR)
4. Open Streamlit:
   - Predictions page
   - Accuracy page
5. Mention limitations + next improvements

