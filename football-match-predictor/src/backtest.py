"""
Backtest — measure model accuracy on past matches

This script takes matches from a given date that already have final results,
predicts each one AS IF we didn't know the score (using only data available
before the match), then immediately compares with the real result.

This lets you see how well the model would have done in real life.

Usage:
    python src/backtest.py                        # defaults to 2026-04-20
    python src/backtest.py --from-date 2026-04-01
    python src/backtest.py --from-date 2026-04-20 --league eng.1
"""

import argparse
from sqlalchemy import text
import pandas as pd

from db import engine
from generate_features import generate_features_for_match
from model_comparison import predict_with_both_models
from predict_upcoming import (
    _primary_prediction,
    build_feature_input,
    create_prediction_run,
    store_prediction,
    predict_scoreline,
)
from player_stats import get_team_availability_impact


OUTCOME_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def run_backtest(from_date: str = "2026-04-20", league_slug: str = "eng.1") -> None:
    """
    For every finished match since `from_date`:
      1. Generate features using only data available before the match
      2. Ask both models to predict the result (they never see the actual score)
      3. Store the prediction in match_predictions (visible in the dashboard)
      4. Immediately reconcile against the real result
      5. Print a summary of how accurate the model was
    """

    # Load all finished matches from the given date
    df = pd.read_sql(
        f"""
        SELECT
            m.id, m.match_date, m.home_team_id, m.away_team_id,
            t1.name AS home_team, t2.name AS away_team,
            m.result, m.home_goals, m.away_goals
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.match_date >= '{from_date}'
          AND m.result IS NOT NULL
        ORDER BY m.match_date ASC
        """,
        engine,
    )

    if df.empty:
        print(f"No finished matches found from {from_date} onwards.")
        print("Make sure you have run: python src/collect_data.py")
        return

    print(f"Found {len(df)} finished matches from {from_date}")
    print("Predicting each one without looking at the result...\n")

    # Create a prediction run so everything is grouped together in the dashboard
    run_id = create_prediction_run(
        league_slug=f"backtest_{from_date.replace('-', '')}",
        total_matches=len(df),
    )

    correct = 0
    availability_cache = {}

    for _, match in df.iterrows():
        # Generate the 28 features using only data before this match's date
        # (same logic as live predictions — no cheating)
        features = generate_features_for_match(
            match_id=match["id"],
            match_date=match["match_date"],
            home_team_id=match["home_team_id"],
            away_team_id=match["away_team_id"],
            form_window=20,
        )
        feature_input = build_feature_input(features)

        # Get RF + LR predictions (models never see the result)
        predictions = predict_with_both_models(feature_input)
        primary = _primary_prediction(predictions)

        # Predict score (adjusting for player availability)
        home_key = (league_slug, match["home_team"])
        away_key = (league_slug, match["away_team"])
        if home_key not in availability_cache:
            availability_cache[home_key] = get_team_availability_impact(league_slug, match["home_team"])
        if away_key not in availability_cache:
            availability_cache[away_key] = get_team_availability_impact(league_slug, match["away_team"])

        scoreline = predict_scoreline(
            feature_input,
            home_penalty=float(availability_cache[home_key].get("availability_penalty", 0.0)),
            away_penalty=float(availability_cache[away_key].get("availability_penalty", 0.0)),
            expected_result=primary,
        )

        # Save the prediction to the database
        store_prediction(
            run_id=run_id,
            match_id=int(match["id"]),
            predictions=predictions,
            predicted_home_goals=scoreline["home_goals"],
            predicted_away_goals=scoreline["away_goals"],
        )

        # Compare with the actual result
        actual = match["result"]
        was_correct = primary == actual
        if was_correct:
            correct += 1

        tick = "✅" if was_correct else "❌"
        rf   = predictions.get("random_forest", {})
        lr   = predictions.get("logistic_regression", {})
        print(
            f"  {tick} {match['match_date'].strftime('%Y-%m-%d')}  "
            f"{match['home_team']} {match['home_goals']}-{match['away_goals']} {match['away_team']}"
            f"\n      Predicted: {OUTCOME_LABELS.get(primary, primary)}  |  "
            f"RF {rf.get('confidence', 0):.0%}  LR {lr.get('confidence', 0):.0%}"
            f"  |  Actual: {OUTCOME_LABELS.get(actual, actual)}"
        )

    # Reconcile all stored predictions against the known results
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE match_predictions mp
                JOIN matches m ON m.id = mp.match_id
                SET
                    mp.actual_home_goals  = m.home_goals,
                    mp.actual_away_goals  = m.away_goals,
                    mp.actual_result      = m.result,
                    mp.outcome_correct    = CASE WHEN mp.predicted_result = m.result THEN 1 ELSE 0 END,
                    mp.score_exact        = CASE
                                               WHEN mp.predicted_home_goals = m.home_goals
                                                AND mp.predicted_away_goals  = m.away_goals THEN 1
                                               ELSE 0
                                           END,
                    mp.status             = 'resolved',
                    mp.resolved_at        = CURRENT_TIMESTAMP
                WHERE mp.run_id = :run_id
                  AND m.result IS NOT NULL
                """
            ),
            {"run_id": run_id},
        )

    # Pull the accuracy for this specific backtest run
    row = engine.connect().execute(
        text(
            """
            SELECT
                COUNT(*)                    AS total,
                SUM(outcome_correct)        AS outcome_correct,
                AVG(outcome_correct) * 100  AS outcome_pct,
                AVG(score_exact) * 100      AS score_pct
            FROM match_predictions
            WHERE run_id = :run_id AND status = 'resolved'
            """
        ),
        {"run_id": run_id},
    ).fetchone()

    total          = int(row[0]) if row else 0
    outcome_correct = int(row[1]) if row and row[1] else 0
    outcome_pct    = float(row[2]) if row and row[2] else 0.0
    score_pct      = float(row[3]) if row and row[3] else 0.0

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS  (from {from_date})")
    print("=" * 60)
    print(f"  Matches predicted   : {total}")
    print(f"  Correct outcomes    : {outcome_correct} / {total}")
    print(f"  Outcome accuracy    : {outcome_pct:.1f}%")
    print(f"  Exact score accuracy: {score_pct:.1f}%")
    print("=" * 60)
    print("\nPredictions are now visible in the dashboard → Predictions tab.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest model on past matches.")
    parser.add_argument(
        "--from-date",
        default="2026-04-20",
        help="Start date for backtest in YYYY-MM-DD format (default: 2026-04-20)",
    )
    parser.add_argument(
        "--league",
        default="eng.1",
        help="ESPN league slug (default: eng.1)",
    )
    args = parser.parse_args()
    run_backtest(from_date=args.from_date, league_slug=args.league)
