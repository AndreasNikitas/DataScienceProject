"""
Train Dual Football Match Prediction Models

This script trains TWO comparison models:
1. Random Forest Classifier (ensemble)
2. Logistic Regression (linear)

This allows side-by-side comparison of model performance and predictions.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from db import engine
from feature_columns import TRAINING_FEATURE_COLUMNS


def load_training_data() -> pd.DataFrame:
    """
    Load match data with engineered features.
    
    Only includes matches with:
    - Known results (finished matches)
    - Generated features (from generate_features.py)
    """
    feature_select = ",\n            ".join(f"mf.{feature}" for feature in TRAINING_FEATURE_COLUMNS)
    query = f"""
        SELECT 
            {feature_select},
            m.result
        FROM match_features mf
        JOIN matches m ON mf.match_id = m.id
        WHERE m.result IS NOT NULL
    """
    return pd.read_sql(query, engine)


def train_and_evaluate_models() -> None:
    """Train both models with proper evaluation."""
    print("Loading training data...")
    df = load_training_data()
    
    if df.empty:
        print("❌ No training data found!")
        print("Run: python src/generate_features.py first")
        return
    
    print(f"Loaded {len(df)} matches with features")
    
    # Define feature columns
    feature_columns = TRAINING_FEATURE_COLUMNS
    
    X = df[feature_columns]
    y = df["result"]
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} matches")
    print(f"Test set: {len(X_test)} matches")
    print(f"Class distribution: {dict(y.value_counts())}")
    
    # ============================================================================
    # MODEL 1: Random Forest
    # ============================================================================
    print("\n" + "="*70)
    print("MODEL 1: RANDOM FOREST CLASSIFIER")
    print("="*70)
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)
    
    print_model_evaluation("Random Forest", y_test, rf_pred, rf_pred_proba, feature_columns, rf_model)
    
    # Save Random Forest model
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    joblib.dump(rf_model, model_path / "random_forest_model.pkl")
    print(f"\n✅ Random Forest model saved to models/random_forest_model.pkl")
    
    # ============================================================================
    # MODEL 2: Logistic Regression
    # ============================================================================
    print("\n" + "="*70)
    print("MODEL 2: LOGISTIC REGRESSION")
    print("="*70)
    
    # Logistic Regression requires feature scaling
    print("Scaling features for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train)
    
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)
    
    print_model_evaluation("Logistic Regression", y_test, lr_pred, lr_pred_proba)
    
    # Save Logistic Regression model and scaler
    joblib.dump(lr_model, model_path / "logistic_regression_model.pkl")
    joblib.dump(scaler, model_path / "feature_scaler.pkl")
    print(f"\n✅ Logistic Regression model saved to models/logistic_regression_model.pkl")
    print(f"✅ Feature scaler saved to models/feature_scaler.pkl")
    
    # ============================================================================
    # MODEL COMPARISON
    # ============================================================================
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    lr_f1 = f1_score(y_test, lr_pred, average='weighted')
    
    print(f"\n{'Metric':<30} {'Random Forest':<20} {'Logistic Reg':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<30} {rf_accuracy:.4f}{'':<14} {lr_accuracy:.4f}")
    print(f"{'Weighted F1':<30} {rf_f1:.4f}{'':<14} {lr_f1:.4f}")
    
    # Determine winner
    winner = "Random Forest" if rf_f1 > lr_f1 else "Logistic Regression"
    print(f"\n🏆 BEST MODEL (by F1): {winner}")
    
    # Save feature importance if available
    try:
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Random Forest Feature Importance:")
        print(feature_importance.to_string(index=False))
        
        # Save to JSON
        feature_importance.to_json(model_path / "feature_importance.json", orient='records')
    except:
        pass
    
    # Save metadata
    metadata = {
        "models": ["random_forest", "logistic_regression"],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_columns,
        "random_forest": {
            "accuracy": float(rf_accuracy),
            "f1_weighted": float(rf_f1)
        },
        "logistic_regression": {
            "accuracy": float(lr_accuracy),
            "f1_weighted": float(lr_f1)
        },
        "best_model": winner.lower().replace(" ", "_")
    }
    
    with open(model_path / "models_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to models/models_metadata.json")
    print("\n" + "="*70)


def print_model_evaluation(model_name: str, y_test, y_pred, y_pred_proba, 
                          feature_columns=None, model=None) -> None:
    """Print comprehensive model evaluation metrics."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n✓ {model_name} Evaluation:")
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Show feature importance for tree-based models
    if feature_columns and model and hasattr(model, 'feature_importances_'):
        print(f"\n  Top 5 Important Features:")
        feature_imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).nlargest(5, 'importance')
        for idx, row in feature_imp.iterrows():
            print(f"    {row['feature']:<30} {row['importance']:.4f}")


if __name__ == "__main__":
    train_and_evaluate_models()
