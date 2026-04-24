"""
Train Football Match Prediction Models

We train two models and compare them:
  1. Random Forest  — good at finding complex patterns
  2. Logistic Regression — simpler, easier to interpret

Improvements over a basic train/test split:
  - GridSearchCV   : automatically finds the best settings for each model
  - Cross-validation: tests accuracy across 5 different splits, not just one
  - Plots          : confusion matrix, feature importance, accuracy over time
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # saves plots as files instead of opening a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, mean_absolute_error,
)

from db import engine
from feature_columns import TRAINING_FEATURE_COLUMNS

# Where to save the plot images
PLOTS_DIR = Path("models/plots")


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """Load all finished matches that have features generated."""
    cols = ", ".join(f"mf.{c}" for c in TRAINING_FEATURE_COLUMNS)
    query = f"""
        SELECT {cols}, m.result, m.home_goals, m.away_goals, m.match_date
        FROM match_features mf
        JOIN matches m ON mf.match_id = m.id
        WHERE m.result IS NOT NULL
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
    """
    return pd.read_sql(query, engine)


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """
    A confusion matrix shows how often the model predicted correctly.
    Rows = actual result, Columns = what the model predicted.
    The diagonal (top-left to bottom-right) should have the biggest numbers.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=["H", "D", "A"])
    labels = ["Home Win", "Draw", "Away Win"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_ylabel("Actual Result")
    ax.set_xlabel("Predicted Result")
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: models/plots/{filename}")


def save_feature_importance(importances, model_name: str) -> None:
    """
    Shows which features the Random Forest relied on most.
    Longer bar = the model pays more attention to that feature.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "feature": TRAINING_FEATURE_COLUMNS,
        "importance": importances,
    }).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df["feature"], df["importance"], color="steelblue")
    ax.set_title(f"{model_name} — Which features matter most?")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved: models/plots/feature_importance.png")


def save_accuracy_over_time(y_true, y_pred, match_dates) -> None:
    """
    Plots how accurate the model was over time on the test set.
    We sort test matches by date and compute a rolling average accuracy.
    If the line drops, the model got worse in that period.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "date": pd.to_datetime(match_dates),
        "correct": (np.array(y_pred) == np.array(y_true)).astype(int),
    }).sort_values("date").reset_index(drop=True)

    # Rolling average over 20 matches to smooth out noise
    df["rolling_acc"] = df["correct"].rolling(20, min_periods=5).mean() * 100
    overall = df["correct"].mean() * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["rolling_acc"], color="steelblue",
            linewidth=2, label="Rolling accuracy (last 20 matches)")
    ax.axhline(overall, color="red", linestyle="--",
               label=f"Overall accuracy ({overall:.1f}%)")
    ax.set_title("Prediction Accuracy Over Time (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(PLOTS_DIR / "accuracy_over_time.png", dpi=150)
    plt.close(fig)
    print("  Saved: models/plots/accuracy_over_time.png")


# ─────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────

def print_results(model_name: str, y_true, y_pred) -> None:
    """Print accuracy and a short classification report."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n  Accuracy : {acc * 100:.2f}%")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Full breakdown:")
    print(classification_report(y_true, y_pred, zero_division=0))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def train_and_evaluate_models() -> None:

    # ── Load data ──────────────────────────────────────────────
    print("Loading data...")
    df = load_training_data()

    if df.empty:
        print("No data found. Run: python src/generate_features.py first")
        return

    print(f"  {len(df)} matches loaded")
    print(f"  Class split: {dict(df['result'].value_counts())}")

    X = df[TRAINING_FEATURE_COLUMNS]
    y = df["result"]

    # 80% for training, 20% for testing
    # stratify=y keeps the H/D/A ratio the same in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    test_dates = df.loc[X_test.index, "match_date"]
    y_goals = df[["home_goals", "away_goals"]]
    y_goals_train = y_goals.loc[X_train.index]
    y_goals_test  = y_goals.loc[X_test.index]

    print(f"\n  Training set : {len(X_train)} matches")
    print(f"  Test set     : {len(X_test)} matches")

    model_path = Path("models")
    model_path.mkdir(exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # MODEL 1 — RANDOM FOREST
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MODEL 1: RANDOM FOREST")
    print("=" * 60)

    # GridSearchCV tries every combination in the grid below and
    # picks the one with the highest cross-validated accuracy.
    print("\nFinding best settings with GridSearchCV...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid={
            "n_estimators": [100, 200],   # number of trees
            "max_depth":    [8, 12],      # how deep each tree can grow
        },
        cv=5,           # 5-fold cross-validation inside the search
        scoring="accuracy",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print(f"  Best settings : {rf_grid.best_params_}")
    print(f"  Best CV score : {rf_grid.best_score_ * 100:.2f}%")

    # Cross-validation: split the data 5 different ways and average the score.
    # This gives a more reliable accuracy than a single train/test split.
    print("\nCross-validation (5 folds)...")
    rf_cv = cross_val_score(
        RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=-1),
        X, y, cv=5, scoring="accuracy",
    )
    print(f"  Accuracy per fold : {[f'{s*100:.1f}%' for s in rf_cv]}")
    print(f"  Average           : {rf_cv.mean()*100:.2f}% ± {rf_cv.std()*100:.2f}%")

    # Train the final model with the best settings
    print("\nTraining final Random Forest...")
    rf_model = RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Calibration makes the confidence percentages more reliable
    rf_calibrated = CalibratedClassifierCV(
        RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=-1),
        method="sigmoid", cv=5,
    )
    rf_calibrated.fit(X_train, y_train)

    rf_pred = rf_calibrated.predict(X_test)
    print_results("Random Forest", y_test, rf_pred)

    print("\nSaving plots...")
    save_confusion_matrix(y_test, rf_pred, "Random Forest")
    save_feature_importance(rf_model.feature_importances_, "Random Forest")
    save_accuracy_over_time(y_test, rf_pred, test_dates)

    joblib.dump(rf_model,     model_path / "random_forest_model.pkl")
    joblib.dump(rf_calibrated, model_path / "random_forest_calibrated_model.pkl")
    print("\n✅ Random Forest saved")

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1       = f1_score(y_test, rf_pred, average="weighted")

    # ═══════════════════════════════════════════════════════════
    # MODEL 2 — LOGISTIC REGRESSION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MODEL 2: LOGISTIC REGRESSION")
    print("=" * 60)

    # Logistic Regression needs features scaled to the same range.
    # We wrap the scaler + model together in a Pipeline so they
    # are always applied in the right order.
    print("\nFinding best settings with GridSearchCV...")
    lr_grid = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(random_state=42, max_iter=2000, solver="lbfgs", n_jobs=-1)),
        ]),
        param_grid={"lr__C": [0.1, 0.5, 1.0, 5.0]},  # C controls regularisation strength
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    best_C = lr_grid.best_params_["lr__C"]
    print(f"  Best C     : {best_C}")
    print(f"  Best score : {lr_grid.best_score_ * 100:.2f}%")

    print("\nCross-validation (5 folds)...")
    lr_cv = cross_val_score(
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best_C, random_state=42, max_iter=2000,
                                      solver="lbfgs", n_jobs=-1)),
        ]),
        X, y, cv=5, scoring="accuracy",
    )
    print(f"  Accuracy per fold : {[f'{s*100:.1f}%' for s in lr_cv]}")
    print(f"  Average           : {lr_cv.mean()*100:.2f}% ± {lr_cv.std()*100:.2f}%")

    print("\nTraining final Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr_model = LogisticRegression(C=best_C, random_state=42, max_iter=2000,
                                   solver="lbfgs", n_jobs=-1)
    lr_model.fit(X_train_scaled, y_train)

    lr_calibrated = CalibratedClassifierCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best_C, random_state=42, max_iter=2000,
                                      solver="lbfgs", n_jobs=-1)),
        ]),
        method="sigmoid", cv=5,
    )
    lr_calibrated.fit(X_train, y_train)

    lr_pred = lr_calibrated.predict(X_test)
    print_results("Logistic Regression", y_test, lr_pred)

    print("\nSaving plots...")
    save_confusion_matrix(y_test, lr_pred, "Logistic Regression")

    joblib.dump(lr_model,    model_path / "logistic_regression_model.pkl")
    joblib.dump(scaler,      model_path / "feature_scaler.pkl")
    joblib.dump(lr_calibrated, model_path / "logistic_regression_calibrated_model.pkl")
    print("\n✅ Logistic Regression saved")

    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_f1       = f1_score(y_test, lr_pred, average="weighted")

    # ═══════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n{'':30} {'Random Forest':>16} {'Logistic Reg':>14}")
    print("-" * 60)
    print(f"{'Test accuracy':30} {rf_accuracy*100:>15.2f}% {lr_accuracy*100:>13.2f}%")
    print(f"{'Weighted F1':30} {rf_f1:>16.4f} {lr_f1:>14.4f}")
    print(f"{'5-fold CV mean':30} {rf_cv.mean()*100:>15.2f}% {lr_cv.mean()*100:>13.2f}%")
    print(f"{'5-fold CV std':30} {rf_cv.std()*100:>15.2f}% {lr_cv.std()*100:>13.2f}%")

    winner = "Random Forest" if rf_f1 > lr_f1 else "Logistic Regression"
    print(f"\n🏆 Best model (by F1): {winner}")

    # ═══════════════════════════════════════════════════════════
    # GOAL MODELS (for score prediction — unchanged)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GOAL MODELS")
    print("=" * 60)

    rf_goals = RandomForestRegressor(n_estimators=200, max_depth=10,
                                      random_state=42, n_jobs=-1)
    rf_goals.fit(X_train, y_goals_train)
    rf_goal_mae = mean_absolute_error(y_goals_test, rf_goals.predict(X_test))
    print(f"\n  Random Forest goal MAE  : {rf_goal_mae:.4f}")
    joblib.dump(rf_goals, model_path / "random_forest_goals_model.pkl")

    lr_goals = LinearRegression()
    lr_goals.fit(X_train_scaled, y_goals_train)
    lr_goal_mae = mean_absolute_error(y_goals_test, lr_goals.predict(X_test_scaled))
    print(f"  Linear Regression MAE   : {lr_goal_mae:.4f}")
    joblib.dump(lr_goals, model_path / "linear_regression_goals_model.pkl")

    # ── Save metadata ───────────────────────────────────────────
    metadata = {
        "best_model": winner.lower().replace(" ", "_"),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": TRAINING_FEATURE_COLUMNS,
        "hyperparameters": {
            "random_forest": rf_grid.best_params_,
            "logistic_regression": {"C": best_C},
        },
        "cross_validation_accuracy": {
            "random_forest":       {"mean": float(rf_cv.mean()), "std": float(rf_cv.std())},
            "logistic_regression": {"mean": float(lr_cv.mean()), "std": float(lr_cv.std())},
        },
        "test_set_accuracy": {
            "random_forest":       float(rf_accuracy),
            "logistic_regression": float(lr_accuracy),
        },
        "goal_model_mae": {
            "random_forest":      float(rf_goal_mae),
            "linear_regression":  float(lr_goal_mae),
        },
    }
    with open(model_path / "models_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata → models/models_metadata.json")
    print(f"✅ Plots    → models/plots/")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    train_and_evaluate_models()
