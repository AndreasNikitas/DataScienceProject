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
