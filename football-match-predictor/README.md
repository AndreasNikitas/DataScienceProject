# football-match-predictor

## Welcome

Welcome to the Football Match Predictor project! This project is developed as part of our coursework to demonstrate data science and web development capabilities.

## Project Overview

The **Football Match Predictor** is a machine learning-based application that predicts football match outcomes using historical match data. The project integrates:

- **Data Collection & Processing**: Gathering and analyzing football match data
- **Machine Learning Model**: Training predictive models to forecast match results
- **Database Management**: Storing and retrieving match data using MariaDB
- **Interactive Dashboard**: A user-friendly Streamlit interface for visualizing predictions and match insights

This project showcases full-stack development skills including Python programming, database design, machine learning implementation, and web-based data visualization.

## Team Members

| Name       |   Surname    | Student ID |
Andreas          Nikitas        25091
Panagiotis       Antoniou       25048
Konstantinos     Christou       25327

---

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

## 7. Run Streamlit dashboard

```powershell
streamlit run dashboard/app.py
```
