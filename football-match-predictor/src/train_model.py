"""
Train Football Match Prediction Model

This script trains a machine learning model to predict match outcomes (H/D/A).
It uses engineered features WITHOUT data leakage.

Key improvements:
1. Uses only pre-match information (team form, not match goals)
2. Train/test split for realistic evaluation
3. Proper evaluation metrics
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from db import engine


def load_training_data() -> pd.DataFrame:
    """
    Load match data with engineered features.
    
    Only includes matches with:
    - Known results (finished matches)
    - Generated features (from generate_features.py)
    """
    query = """
        SELECT 
            mf.home_last5_points,
            mf.home_last5_goal_diff,
            mf.home_form_pct,
            mf.away_last5_points,
            mf.away_last5_goal_diff,
            mf.away_form_pct,
            mf.h2h_home_wins,
            mf.h2h_draws,
            mf.h2h_away_wins,
            m.result
        FROM match_features mf
        JOIN matches m ON mf.match_id = m.id
        WHERE m.result IS NOT NULL
    """
    return pd.read_sql(query, engine)


def train_and_evaluate_model() -> None:
    """Train model with proper evaluation."""
    print("Loading training data...")
    df = load_training_data()
    
    if df.empty:
        print("❌ No training data found!")
        print("Run: python src/generate_features.py first")
        return
    
    print(f"Loaded {len(df)} matches with features")
    
    # Define feature columns (NO goals or results from the match itself!)
    feature_columns = [
        "home_last5_points",
        "home_last5_goal_diff",
        "home_form_pct",
        "away_last5_points",
        "away_last5_goal_diff",
        "away_form_pct",
        "h2h_home_wins",
        "h2h_draws",
        "h2h_away_wins"
    ]
    
    X = df[feature_columns]
    y = df["result"]
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} matches")
    print(f"Test set: {len(X_test)} matches")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))
    
    print(f"\nConfusion Matrix:")
    print(f"             Predicted")
    print(f"Actual    A    D    H")
    cm = confusion_matrix(y_test, y_pred, labels=['A', 'D', 'H'])
    for i, label in enumerate(['A', 'D', 'H']):
        print(f"  {label}     {cm[i][0]:3d}  {cm[i][1]:3d}  {cm[i][2]:3d}")
    
    # Feature importance
    print(f"\nTop 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    train_and_evaluate_model()
