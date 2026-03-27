"""
Make predictions for upcoming matches

This demonstrates how to use the trained model for real predictions.
"""

import joblib
import pandas as pd
from pathlib import Path

from db import engine
from generate_features import generate_features_for_match


def load_model():
    """Load the trained model."""
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "model.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Run: python src/train_model.py first"
        )
    
    return joblib.load(model_path)


def predict_upcoming_matches():
    """Predict results for all upcoming matches."""
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
        ORDER BY m.match_date ASC
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No upcoming matches found!")
        return
    
    print(f"Found {len(df)} upcoming matches\n")
    print("="*80)
    
    # Load model
    model = load_model()
    
    # Make predictions for each match
    for _, match in df.iterrows():
        # Generate features for this match
        features = generate_features_for_match(
            match['id'],
            match['match_date'],
            match['home_team_id'],
            match['away_team_id']
        )
        
        # Prepare feature vector (same order as training)
        X = pd.DataFrame([{
            'home_last5_points': features['home_last5_points'],
            'home_last5_goal_diff': features['home_last5_goal_diff'],
            'home_form_pct': features['home_form_pct'],
            'away_last5_points': features['away_last5_points'],
            'away_last5_goal_diff': features['away_last5_goal_diff'],
            'away_form_pct': features['away_form_pct'],
            'h2h_home_wins': features['h2h_home_wins'],
            'h2h_draws': features['h2h_draws'],
            'h2h_away_wins': features['h2h_away_wins']
        }])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map prediction to outcome
        outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        
        # Get probability index
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        print(f"\n📅 {match['match_date'].strftime('%Y-%m-%d %H:%M')}")
        print(f"   {match['home_team']} vs {match['away_team']}")
        print(f"\n   Prediction: {outcome_map[prediction]} ({prediction})")
        print(f"\n   Probabilities:")
        print(f"      Home Win (H): {prob_dict.get('H', 0):.1%}")
        print(f"      Draw (D):     {prob_dict.get('D', 0):.1%}")
        print(f"      Away Win (A): {prob_dict.get('A', 0):.1%}")
        print(f"\n   Team Form:")
        print(f"      {match['home_team']}: {features['home_last5_points']} pts (last 5), "
              f"{features['home_form_pct']:.1f}% form")
        print(f"      {match['away_team']}: {features['away_last5_points']} pts (last 5), "
              f"{features['away_form_pct']:.1f}% form")
        print("-"*80)
    
    print("\n✅ Predictions complete!")
    print("\nNote: These are probabilistic predictions based on recent form.")
    print("Football is unpredictable - use these as guidance, not guarantees!")


if __name__ == "__main__":
    predict_upcoming_matches()
