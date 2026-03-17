from pathlib import Path

import joblib
import pandas as pd


def load_model():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py first.")

    return joblib.load(model_path)


def predict_example() -> None:
    model = load_model()

    # TODO: Replace this sample input with real match inputs from your app.
    sample_input = pd.DataFrame(
        [
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "home_goals": 0,
                "away_goals": 0,
            }
        ]
    )

    prediction = model.predict(sample_input)[0]
    print(f"Predicted result: {prediction}")


if __name__ == "__main__":
    predict_example()
