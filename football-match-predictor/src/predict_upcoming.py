"""
Make predictions for upcoming matches

Uses BOTH trained models (Random Forest + Logistic Regression) for predictions.
This allows side-by-side comparison and consensus predictions.
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
from sqlalchemy import text

from db import engine
from feature_columns import TRAINING_FEATURE_COLUMNS
from generate_features import calculate_team_form, generate_features_for_match
from model_comparison import predict_with_both_models
from player_stats import get_team_availability_impact


def load_models():
    """Load both trained models."""
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    
    models = {}
    
    # Try to load Random Forest
    rf_path = models_dir / "random_forest_model.pkl"
    if rf_path.exists():
        models['random_forest'] = joblib.load(rf_path)
    
    # Try to load Logistic Regression
    lr_path = models_dir / "logistic_regression_model.pkl"
    if lr_path.exists():
        models['logistic_regression'] = joblib.load(lr_path)
        scaler_path = models_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            models['scaler'] = joblib.load(scaler_path)
    
    if not models:
        raise FileNotFoundError(
            f"No models found in {models_dir}\n"
            "Run: python src/train_models.py first"
        )
    
    return models


def _derive_result_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def _clamp_score(value: float) -> int:
    return max(0, min(6, int(round(value))))


def _get_goal_baseline() -> Tuple[float, float]:
    query = text(
        """
        SELECT AVG(home_goals), AVG(away_goals)
        FROM matches
        WHERE result IS NOT NULL
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).fetchone()

    if not row:
        return 1.4, 1.1
    home_avg = float(row[0]) if row[0] is not None else 1.4
    away_avg = float(row[1]) if row[1] is not None else 1.1
    return home_avg, away_avg


def estimate_scoreline(
    match_date: datetime,
    home_team_id: int,
    away_team_id: int,
    home_penalty: float = 0.0,
    away_penalty: float = 0.0,
) -> Dict[str, int]:
    home_form = calculate_team_form(home_team_id, match_date, num_matches=10)
    away_form = calculate_team_form(away_team_id, match_date, num_matches=10)
    baseline_home, baseline_away = _get_goal_baseline()

    home_matches = max(home_form["matches_played"], 1)
    away_matches = max(away_form["matches_played"], 1)

    home_attack = home_form["goals_scored"] / home_matches if home_form["matches_played"] else baseline_home
    home_defense = home_form["goals_conceded"] / home_matches if home_form["matches_played"] else baseline_away
    away_attack = away_form["goals_scored"] / away_matches if away_form["matches_played"] else baseline_away
    away_defense = away_form["goals_conceded"] / away_matches if away_form["matches_played"] else baseline_home

    expected_home_goals = ((home_attack + away_defense) / 2) * 1.08
    expected_away_goals = ((away_attack + home_defense) / 2) * 0.95

    expected_home_goals = expected_home_goals * (1 - max(0.0, min(home_penalty, 0.35)))
    expected_away_goals = expected_away_goals * (1 - max(0.0, min(away_penalty, 0.35)))

    return {
        "home_goals": _clamp_score(expected_home_goals),
        "away_goals": _clamp_score(expected_away_goals),
    }


def _primary_prediction(predictions: Dict) -> str:
    consensus = predictions.get("consensus")
    if consensus and consensus.get("agreement"):
        return consensus["prediction"]

    candidates = []
    rf = predictions.get("random_forest")
    lr = predictions.get("logistic_regression")
    if rf:
        candidates.append((rf["confidence"], rf["prediction"]))
    if lr:
        candidates.append((lr["confidence"], lr["prediction"]))
    if not candidates:
        raise ValueError("No model predictions available.")
    return max(candidates, key=lambda c: c[0])[1]


def create_prediction_run(league_slug: str, total_matches: int) -> int:
    query = text(
        """
        INSERT INTO prediction_runs (league_slug, total_matches)
        VALUES (:league_slug, :total_matches)
        """
    )
    with engine.begin() as conn:
        result = conn.execute(query, {"league_slug": league_slug, "total_matches": total_matches})
        return int(result.lastrowid)


def store_prediction(
    run_id: int,
    match_id: int,
    predictions: Dict,
    predicted_home_goals: int,
    predicted_away_goals: int,
) -> None:
    primary_prediction = _primary_prediction(predictions)
    expected_result = _derive_result_from_score(predicted_home_goals, predicted_away_goals)
    if primary_prediction != expected_result:
        primary_prediction = expected_result

    rf = predictions.get("random_forest")
    lr = predictions.get("logistic_regression")
    consensus = predictions.get("consensus")

    query = text(
        """
        INSERT INTO match_predictions (
            run_id, match_id, predicted_result, predicted_home_goals, predicted_away_goals,
            rf_prediction, rf_confidence, lr_prediction, lr_confidence, consensus_prediction
        )
        VALUES (
            :run_id, :match_id, :predicted_result, :predicted_home_goals, :predicted_away_goals,
            :rf_prediction, :rf_confidence, :lr_prediction, :lr_confidence, :consensus_prediction
        )
        ON DUPLICATE KEY UPDATE
            predicted_result = VALUES(predicted_result),
            predicted_home_goals = VALUES(predicted_home_goals),
            predicted_away_goals = VALUES(predicted_away_goals),
            rf_prediction = VALUES(rf_prediction),
            rf_confidence = VALUES(rf_confidence),
            lr_prediction = VALUES(lr_prediction),
            lr_confidence = VALUES(lr_confidence),
            consensus_prediction = VALUES(consensus_prediction),
            updated_at = CURRENT_TIMESTAMP
        """
    )
    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "run_id": run_id,
                "match_id": match_id,
                "predicted_result": primary_prediction,
                "predicted_home_goals": predicted_home_goals,
                "predicted_away_goals": predicted_away_goals,
                "rf_prediction": rf["prediction"] if rf else None,
                "rf_confidence": rf["confidence"] if rf else None,
                "lr_prediction": lr["prediction"] if lr else None,
                "lr_confidence": lr["confidence"] if lr else None,
                "consensus_prediction": (
                    consensus["prediction"]
                    if consensus and consensus.get("prediction") in {"H", "D", "A"}
                    else None
                ),
            },
        )


def reconcile_resolved_predictions() -> int:
    query = text(
        """
        UPDATE match_predictions mp
        JOIN matches m ON m.id = mp.match_id
        SET
            mp.actual_home_goals = m.home_goals,
            mp.actual_away_goals = m.away_goals,
            mp.actual_result = m.result,
            mp.outcome_correct = CASE WHEN mp.predicted_result = m.result THEN 1 ELSE 0 END,
            mp.score_exact = CASE
                WHEN mp.predicted_home_goals = m.home_goals
                 AND mp.predicted_away_goals = m.away_goals THEN 1
                ELSE 0
            END,
            mp.status = 'resolved',
            mp.resolved_at = CURRENT_TIMESTAMP
        WHERE mp.status = 'pending'
          AND m.result IS NOT NULL
        """
    )
    with engine.begin() as conn:
        result = conn.execute(query)
        return int(result.rowcount or 0)


def get_accuracy_summary() -> Dict[str, float]:
    query = text(
        """
        SELECT
            COUNT(*) AS total_resolved,
            AVG(outcome_correct) * 100 AS outcome_accuracy_pct,
            AVG(score_exact) * 100 AS exact_score_accuracy_pct
        FROM match_predictions
        WHERE status = 'resolved'
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).fetchone()

    if not row or row[0] == 0:
        return {"total_resolved": 0, "outcome_accuracy_pct": 0.0, "exact_score_accuracy_pct": 0.0}
    return {
        "total_resolved": int(row[0]),
        "outcome_accuracy_pct": float(row[1] or 0.0),
        "exact_score_accuracy_pct": float(row[2] or 0.0),
    }


def build_feature_input(features: Dict) -> Dict:
    return {column: float(features.get(column, 0.0)) for column in TRAINING_FEATURE_COLUMNS}


def predict_upcoming_matches(league_slug: str = "eng.1", form_window: int = 9):
    """Predict results for all upcoming matches using both models."""
    reconciled = reconcile_resolved_predictions()
    if reconciled:
        print(f"Updated {reconciled} stored predictions with final scores.")

    # Get upcoming matches (no result yet)
    query = """
        SELECT 
            m.id,
            m.match_date,
            m.home_team_id,
            m.away_team_id,
            t1.name as home_team,
            t2.name as away_team
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.result IS NULL
          AND m.match_date >= UTC_TIMESTAMP()
        ORDER BY m.match_date ASC
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No upcoming matches found!")
        return
    
    print(f"Found {len(df)} upcoming matches\n")
    print("="*100)
    
    # Load models
    models = load_models()
    print(f"✅ Loaded models: {', '.join(m for m in models.keys() if m != 'scaler')}\n")
    
    predictions_list = []
    run_id = create_prediction_run(league_slug, len(df))
    availability_cache: Dict[Tuple[str, str], Dict] = {}
    
    # Make predictions for each match
    for _, match in df.iterrows():
        # Generate features for this match
        features = generate_features_for_match(
            match['id'],
            match['match_date'],
            match['home_team_id'],
            match['away_team_id'],
            form_window=form_window,
        )
        
        feature_input = build_feature_input(features)
        
        # Make predictions with both models
        predictions = predict_with_both_models(feature_input)
        home_team_name = str(match["home_team"])
        away_team_name = str(match["away_team"])
        home_cache_key = (league_slug, home_team_name)
        away_cache_key = (league_slug, away_team_name)

        if home_cache_key not in availability_cache:
            availability_cache[home_cache_key] = get_team_availability_impact(league_slug, home_team_name)
        if away_cache_key not in availability_cache:
            availability_cache[away_cache_key] = get_team_availability_impact(league_slug, away_team_name)

        home_availability = availability_cache[home_cache_key]
        away_availability = availability_cache[away_cache_key]

        scoreline = estimate_scoreline(
            match["match_date"],
            int(match["home_team_id"]),
            int(match["away_team_id"]),
            home_penalty=float(home_availability.get("availability_penalty", 0.0)),
            away_penalty=float(away_availability.get("availability_penalty", 0.0)),
        )
        store_prediction(
            run_id=run_id,
            match_id=int(match["id"]),
            predictions=predictions,
            predicted_home_goals=scoreline["home_goals"],
            predicted_away_goals=scoreline["away_goals"],
        )
        
        # Print match prediction
        print(f"\n📅 {match['match_date'].strftime('%Y-%m-%d %H:%M')}")
        print(f"   🏆 {match['home_team']} vs {match['away_team']}")
        print("   " + "-"*80)
        
        outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        
        # Random Forest prediction
        if 'random_forest' in predictions:
            rf = predictions['random_forest']
            print(f"   🌲 Random Forest: {outcome_map[rf['prediction']]} "
                  f"(confidence: {rf['confidence']:.1%})")
        
        # Logistic Regression prediction
        if 'logistic_regression' in predictions:
            lr = predictions['logistic_regression']
            print(f"   📈 Logistic Reg:  {outcome_map[lr['prediction']]} "
                  f"(confidence: {lr['confidence']:.1%})")
        
        # Consensus
        if 'consensus' in predictions:
            cons = predictions['consensus']
            status = "✓ MODELS AGREE" if cons.get('agreement') else "✗ MODELS DISAGREE"
            if cons.get("prediction") in outcome_map:
                print(f"   🎯 Consensus: {outcome_map[cons['prediction']]} {status}")
            else:
                print(f"   🎯 Consensus: Models disagree {status}")

        print(
            f"   ⚽ Predicted score: {match['home_team']} {scoreline['home_goals']} - "
            f"{scoreline['away_goals']} {match['away_team']}"
        )
        
        # Team form
        print(f"\n   📊 Team Form:")
        print(f"      {match['home_team']}: {features['home_last5_points']} pts (last 5), "
              f"{features['home_form_pct']:.1f}% form")
        print(f"      {match['away_team']}: {features['away_last5_points']} pts (last 5), "
              f"{features['away_form_pct']:.1f}% form")
        print(
            "\n   👤 Player Availability Impact:"
            f"\n      {home_team_name}: penalty {home_availability.get('availability_penalty', 0.0):.1%},"
            f" injuries {home_availability.get('injured_players', 0)},"
            f" key absences {', '.join(home_availability.get('key_absences', [])) or 'none'}"
            f"\n      {away_team_name}: penalty {away_availability.get('availability_penalty', 0.0):.1%},"
            f" injuries {away_availability.get('injured_players', 0)},"
            f" key absences {', '.join(away_availability.get('key_absences', [])) or 'none'}"
        )
        
        predictions_list.append({
            'match_id': match['id'],
            'date': match['match_date'].isoformat(),
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'predictions': predictions,
            'predicted_score': {
                'home_goals': scoreline['home_goals'],
                'away_goals': scoreline['away_goals']
            },
            'player_availability': {
                'home': home_availability,
                'away': away_availability
            },
        })
    
    print("\n" + "="*100)
    print("\n✅ Predictions complete!")
    print("\nNote: These are probabilistic predictions based on recent form.")
    print("Football is unpredictable - use these as guidance, not guarantees!")
    print(f"\nPrediction run id: {run_id}")
    
    # Save predictions to file
    save_predictions(predictions_list)

    summary = get_accuracy_summary()
    if summary["total_resolved"] > 0:
        print(
            "\n📊 Stored prediction accuracy "
            f"(resolved: {int(summary['total_resolved'])}): "
            f"Outcome {summary['outcome_accuracy_pct']:.1f}% | "
            f"Exact score {summary['exact_score_accuracy_pct']:.1f}%"
        )
    else:
        print("\n📊 Stored prediction accuracy: no resolved predictions yet.")
    
    return predictions_list


def save_predictions(predictions: List[Dict]) -> None:
    """Save predictions to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_predictions': len(predictions),
        'predictions': predictions
    }
    
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Predictions saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict upcoming matches.")
    parser.add_argument(
        "--league",
        default="eng.1",
        help="ESPN league slug used for player availability stats (eng.1, esp.1, etc.)",
    )
    parser.add_argument(
        "--form-window",
        type=int,
        default=9,
        help="Recent-form window used for feature generation (recommended: 9)",
    )

    args = parser.parse_args()
    
    predict_upcoming_matches(league_slug=args.league, form_window=args.form_window)
