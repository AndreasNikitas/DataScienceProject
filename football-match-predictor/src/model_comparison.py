"""
Model Comparison and Evaluation Framework

Provides functionality to:
- Load trained models
- Compare predictions from multiple models
- Evaluate agreement/disagreement between models
- Generate comparison reports
- Track model performance over time
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from db import engine


class ModelComparator:
    """Compare multiple trained models on same dataset."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize with path to saved models."""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.metadata = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load saved models from disk."""
        try:
            self.models['random_forest'] = joblib.load(
                self.models_dir / "random_forest_model.pkl"
            )
            print("✅ Random Forest model loaded")
        except FileNotFoundError:
            print("⚠️  Random Forest model not found")
        
        try:
            self.models['logistic_regression'] = joblib.load(
                self.models_dir / "logistic_regression_model.pkl"
            )
            self.scaler = joblib.load(self.models_dir / "feature_scaler.pkl")
            print("✅ Logistic Regression model loaded")
        except FileNotFoundError:
            print("⚠️  Logistic Regression model not found")
        
        try:
            with open(self.models_dir / "models_metadata.json") as f:
                self.metadata = json.load(f)
            print("✅ Model metadata loaded")
        except FileNotFoundError:
            print("⚠️  Model metadata not found")
    
    def get_predictions_on_test_set(self) -> Dict:
        """Get predictions from all models on test set."""
        # Load test data
        query = """
            SELECT 
                m.id,
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
            LIMIT 100
        """
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return {"error": "No test data available"}
        
        feature_columns = [col for col in df.columns if col not in ['id', 'result']]
        X = df[feature_columns]
        y = df['result']
        
        results = {
            'test_data': df,
            'X': X,
            'y': y,
            'predictions': {}
        }
        
        # Random Forest predictions
        if 'random_forest' in self.models:
            rf_pred = self.models['random_forest'].predict(X)
            rf_proba = self.models['random_forest'].predict_proba(X)
            results['predictions']['random_forest'] = {
                'predictions': rf_pred,
                'probabilities': rf_proba
            }
        
        # Logistic Regression predictions (needs scaling)
        if 'logistic_regression' in self.models and self.scaler:
            X_scaled = self.scaler.transform(X)
            lr_pred = self.models['logistic_regression'].predict(X_scaled)
            lr_proba = self.models['logistic_regression'].predict_proba(X_scaled)
            results['predictions']['logistic_regression'] = {
                'predictions': lr_pred,
                'probabilities': lr_proba
            }
        
        return results
    
    def compare_model_agreement(self, predictions: Dict) -> Dict:
        """Analyze agreement between model predictions."""
        if 'predictions' not in predictions:
            return {"error": "Invalid predictions format"}
        
        preds = predictions['predictions']
        y_true = predictions['y']
        
        if len(preds) < 2:
            return {"error": "Need at least 2 models to compare"}
        
        model_names = list(preds.keys())
        comparison = {
            'total_samples': len(y_true),
            'model_names': model_names,
            'individual_accuracy': {},
            'pairwise_agreement': {},
            'consensus_analysis': {}
        }
        
        # Individual accuracy
        for model_name, pred_data in preds.items():
            acc = accuracy_score(y_true, pred_data['predictions'])
            comparison['individual_accuracy'][model_name] = acc
        
        # Pairwise agreement
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                pred1 = preds[model1]['predictions']
                pred2 = preds[model2]['predictions']
                agreement = np.mean(pred1 == pred2)
                comparison['pairwise_agreement'][f"{model1}_vs_{model2}"] = agreement
        
        # Consensus analysis
        all_preds = np.array([preds[m]['predictions'] for m in model_names])
        consensus = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, all_preds)
        
        consensus_acc = accuracy_score(y_true, consensus)
        consensus_disagreement = np.mean(
            ~np.all(all_preds == all_preds[0], axis=0)
        )
        
        comparison['consensus_analysis'] = {
            'consensus_accuracy': consensus_acc,
            'disagreement_rate': consensus_disagreement,
            'consensus_predictions': consensus.tolist()
        }
        
        return comparison
    
    def generate_comparison_report(self) -> None:
        """Generate and print comprehensive model comparison report."""
        print("\n" + "="*70)
        print("MODEL COMPARISON REPORT")
        print("="*70)
        
        if not self.models:
            print("❌ No models loaded")
            return
        
        print(f"\n📦 Loaded Models: {list(self.models.keys())}")
        
        if self.metadata:
            print(f"📊 Best Model: {self.metadata.get('best_model', 'Unknown')}")
        
        # Get predictions
        predictions = self.get_predictions_on_test_set()
        
        if 'error' in predictions:
            print(f"⚠️  {predictions['error']}")
            return
        
        # Analyze agreement
        comparison = self.compare_model_agreement(predictions)
        
        if 'error' in comparison:
            print(f"⚠️  {comparison['error']}")
            return
        
        print(f"\n📈 INDIVIDUAL ACCURACY")
        for model, acc in comparison['individual_accuracy'].items():
            print(f"  {model:<30} {acc:.4f} ({acc*100:.2f}%)")
        
        print(f"\n🤝 PAIRWISE AGREEMENT")
        for pair, agreement in comparison['pairwise_agreement'].items():
            print(f"  {pair:<40} {agreement:.4f} ({agreement*100:.2f}%)")
        
        print(f"\n🎯 CONSENSUS ANALYSIS")
        consensus = comparison['consensus_analysis']
        print(f"  Consensus Accuracy: {consensus['consensus_accuracy']:.4f}")
        print(f"  Disagreement Rate:  {consensus['disagreement_rate']:.4f}")
        
        print("\n" + "="*70)


def load_model_for_prediction(model_name: str) -> Tuple[Optional[object], Optional[object]]:
    """
    Load a single model for prediction use.
    
    Returns:
        Tuple of (model, scaler) - scaler is None for non-linear models
    """
    models_dir = Path("models")
    
    if model_name == "random_forest":
        calibrated_path = models_dir / "random_forest_calibrated_model.pkl"
        if calibrated_path.exists():
            model = joblib.load(calibrated_path)
            return model, None
        model = joblib.load(models_dir / "random_forest_model.pkl")
        return model, None
    
    elif model_name == "logistic_regression":
        calibrated_path = models_dir / "logistic_regression_calibrated_model.pkl"
        if calibrated_path.exists():
            model = joblib.load(calibrated_path)
            return model, None
        model = joblib.load(models_dir / "logistic_regression_model.pkl")
        scaler = joblib.load(models_dir / "feature_scaler.pkl")
        return model, scaler
    
    return None, None


def predict_with_both_models(features_dict: Dict) -> Dict:
    """
    Make predictions using both models on new data.
    
    Args:
        features_dict: Dict with feature names and values
    
    Returns:
        Dict with predictions from both models and confidence scores
    """
    import pandas as pd
    
    # Convert to DataFrame (single row)
    X = pd.DataFrame([features_dict])
    
    predictions = {}
    
    # Random Forest
    try:
        rf_model, _ = load_model_for_prediction("random_forest")
        if rf_model:
            rf_pred = rf_model.predict(X)[0]
            rf_proba = rf_model.predict_proba(X)[0]
            predictions['random_forest'] = {
                'prediction': rf_pred,
                'confidence': float(np.max(rf_proba))
            }
    except Exception as e:
        print(f"⚠️  Error with Random Forest: {e}")
    
    # Logistic Regression
    try:
        lr_model, scaler = load_model_for_prediction("logistic_regression")
        if lr_model:
            if scaler:
                X_input = scaler.transform(X)
            else:
                X_input = X
            lr_pred = lr_model.predict(X_input)[0]
            lr_proba = lr_model.predict_proba(X_input)[0]
            predictions['logistic_regression'] = {
                'prediction': lr_pred,
                'confidence': float(np.max(lr_proba))
            }
    except Exception as e:
        print(f"⚠️  Error with Logistic Regression: {e}")
    
    # Consensus
    if len(predictions) == 2:
        rf_pred = predictions['random_forest']['prediction']
        lr_pred = predictions['logistic_regression']['prediction']
        predictions['consensus'] = {
            'prediction': rf_pred if rf_pred == lr_pred else 'DISAGREEMENT',
            'agreement': rf_pred == lr_pred
        }
    
    return predictions


def predict_goal_scores_with_both_models(features_dict: Dict) -> Dict:
    """
    Predict home/away goals using both goal models and return an ensemble estimate.
    """
    import pandas as pd

    X = pd.DataFrame([features_dict])
    predictions = {}
    models_dir = Path("models")

    try:
        rf_goals_model = joblib.load(models_dir / "random_forest_goals_model.pkl")
        rf_pred = rf_goals_model.predict(X)[0]
        predictions["random_forest"] = {
            "home_goals": float(rf_pred[0]),
            "away_goals": float(rf_pred[1]),
        }
    except Exception as e:
        print(f"⚠️  Error with Random Forest goal model: {e}")

    try:
        linear_goals_model = joblib.load(models_dir / "linear_regression_goals_model.pkl")
        scaler = joblib.load(models_dir / "feature_scaler.pkl")
        X_scaled = scaler.transform(X)
        lr_pred = linear_goals_model.predict(X_scaled)[0]
        predictions["linear_regression"] = {
            "home_goals": float(lr_pred[0]),
            "away_goals": float(lr_pred[1]),
        }
    except Exception as e:
        print(f"⚠️  Error with Linear Regression goal model: {e}")

    goal_candidates = [
        model_prediction
        for model_prediction in predictions.values()
        if "home_goals" in model_prediction and "away_goals" in model_prediction
    ]
    if goal_candidates:
        predictions["ensemble"] = {
            "home_goals": float(np.mean([p["home_goals"] for p in goal_candidates])),
            "away_goals": float(np.mean([p["away_goals"] for p in goal_candidates])),
        }

    return predictions


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.generate_comparison_report()
