# football-match-predictor

Minimal starter project using Python, MariaDB, and Streamlit.

## 1. Start XAMPP (MariaDB)
- Open XAMPP Control Panel
- Start **MySQL**

## 2. Create and activate virtual environment
From the project root:

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Set environment variables
- Copy `.env.example` to `.env`
- Update values if needed

Default values:
- DB_HOST=127.0.0.1
- DB_PORT=3306
- DB_NAME=football_predictor
- DB_USER=root
- DB_PASSWORD=

## 5. Initialize database schema
This reads `sql/schema.sql` and creates the database/tables.

```powershell
python src/init_db.py
```

## 6. Optional: Train starter model

```powershell
python src/train_model.py
```

For dual-model predictions (recommended):

```powershell
python src/train_models.py
```

## Data Cleaning and Quality Constraints
- Ingestion now cleans and validates API data in `src/collect_data.py`.
- For an existing database (already created before these changes), run:

```powershell
mysql -u root football_predictor < sql/data_quality_migration.sql
```

- Then collect fresh data:

```powershell
python src/collect_data.py
```

## 7. Run Streamlit dashboard

```powershell
streamlit run dashboard/app.py

```

## 8. Predict upcoming matches and keep accuracy history

```powershell
python src/predict_upcoming.py
```

This now:
- Saves each upcoming match prediction in MySQL (including predicted scoreline)
- Keeps it until final score is available
- Automatically marks resolved predictions and stores:
  - outcome accuracy (H/D/A correct)
  - exact-score accuracy
- Uses the two outcome models for `H/D/A` prediction
- Uses two goal models to estimate the scoreline, then applies availability penalties from ESPN roster stats/injuries

## 9. Player stats / availability (injuries)

CLI:

```powershell
python src/player_stats.py --league esp.1 --team Barcelona
```

Dashboard:
- Open **👤 Player Stats**
- Select league + team
- View squad stats and unavailable/injury list (when present in ESPN feed)

To include player availability in predictions:

```powershell
python src/predict_upcoming.py --league esp.1
```

## 10. Validate "last N matches" window (3/5/8/10)

```powershell
python src/validate_form_windows.py --windows 5 8 10 15 20
```

This prints train/test accuracy for both models by window and saves:
- `models/form_window_validation_repeated.json`

To repeat across many random seeds:

```powershell
python src/validate_form_windows.py --windows 3 5 8 10 --seeds 1 2 3 4 5 6 7 8 9 10
```

Current historical training features now include:
- Form + H2H (existing)
- Rest/fatigue (days since last match, matches in last 7/14 days)
- Travel pressure proxy (home/away penalties from short rest and away streaks)
- Home/away long-window strength (rolling PPG and goal diff)
- Rolling Elo before each match (home, away, and Elo diff)

Note: historical injury snapshots are not available in the current ESPN source, so injury impact is not part of model training.
