from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from db import engine


def load_match_data() -> pd.DataFrame:
    query = """
        SELECT home_team_id, away_team_id, home_goals, away_goals, result
        FROM matches
        WHERE result IS NOT NULL
    """
    return pd.read_sql(query, engine)


def build_placeholder_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty:
        return df

    print("No match data found. Using a tiny placeholder dataset.")
    return pd.DataFrame(
        [
            {"home_team_id": 1, "away_team_id": 2, "home_goals": 2, "away_goals": 1, "result": "H"},
            {"home_team_id": 2, "away_team_id": 3, "home_goals": 1, "away_goals": 1, "result": "D"},
            {"home_team_id": 3, "away_team_id": 1, "home_goals": 0, "away_goals": 2, "result": "A"},
        ]
    )


def train_and_save_model() -> None:
    raw_df = load_match_data()
    df = build_placeholder_dataset(raw_df)

    # TODO: Add better feature engineering here.
    feature_columns = ["home_team_id", "away_team_id", "home_goals", "away_goals"]
    X = df[feature_columns].fillna(0)
    y = df["result"]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()
