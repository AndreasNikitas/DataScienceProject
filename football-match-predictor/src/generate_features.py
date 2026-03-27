"""
Feature Engineering for Football Match Prediction

This script calculates predictive features from historical match data.
These features are used for machine learning WITHOUT data leakage.

Key Concept: We only use information available BEFORE a match starts.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import text

from db import engine


def calculate_team_form(team_id: int, before_date: datetime, num_matches: int = 5) -> Dict:
    """
    Calculate team's recent form statistics.
    
    Args:
        team_id: Team to calculate for
        before_date: Calculate form using matches BEFORE this date
        num_matches: Number of recent matches to consider (default 5)
    
    Returns:
        Dict with wins, draws, losses, points, goals_scored, goals_conceded
    """
    # Get last N matches for this team BEFORE the specified date
    query = text("""
        SELECT 
            CASE 
                WHEN home_team_id = :team_id THEN 'home'
                ELSE 'away'
            END as venue,
            home_goals,
            away_goals,
            result
        FROM matches
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND match_date < :before_date
          AND result IS NOT NULL
        ORDER BY match_date DESC
        LIMIT :num_matches
    """)
    
    with engine.connect() as conn:
        results = conn.execute(
            query, 
            {"team_id": team_id, "before_date": before_date, "num_matches": num_matches}
        ).fetchall()
    
    if not results:
        return {
            "wins": 0, "draws": 0, "losses": 0, "points": 0,
            "goals_scored": 0, "goals_conceded": 0, "goal_diff": 0,
            "matches_played": 0
        }
    
    wins = draws = losses = 0
    goals_scored = goals_conceded = 0
    
    for row in results:
        venue, home_goals, away_goals, result = row
        
        if venue == 'home':
            goals_scored += home_goals
            goals_conceded += away_goals
            if result == 'H':
                wins += 1
            elif result == 'D':
                draws += 1
            else:
                losses += 1
        else:  # away
            goals_scored += away_goals
            goals_conceded += home_goals
            if result == 'A':
                wins += 1
            elif result == 'D':
                draws += 1
            else:
                losses += 1
    
    points = wins * 3 + draws * 1
    
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "goal_diff": goals_scored - goals_conceded,
        "matches_played": len(results)
    }


def calculate_head_to_head(home_team_id: int, away_team_id: int, before_date: datetime) -> Dict:
    """
    Calculate head-to-head statistics between two teams.
    
    Args:
        home_team_id: Home team
        away_team_id: Away team
        before_date: Only consider matches before this date
    
    Returns:
        Dict with home_wins, draws, away_wins
    """
    query = text("""
        SELECT result
        FROM matches
        WHERE home_team_id = :home_id 
          AND away_team_id = :away_id
          AND match_date < :before_date
          AND result IS NOT NULL
    """)
    
    with engine.connect() as conn:
        results = conn.execute(
            query,
            {"home_id": home_team_id, "away_id": away_team_id, "before_date": before_date}
        ).fetchall()
    
    home_wins = sum(1 for r in results if r[0] == 'H')
    draws = sum(1 for r in results if r[0] == 'D')
    away_wins = sum(1 for r in results if r[0] == 'A')
    
    return {
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins
    }


def generate_features_for_match(match_id: int, match_date: datetime, 
                                 home_team_id: int, away_team_id: int) -> Dict:
    """
    Generate all features for a single match.
    
    This is the main function that combines all feature calculations.
    """
    # Calculate home team form
    home_form = calculate_team_form(home_team_id, match_date)
    
    # Calculate away team form
    away_form = calculate_team_form(away_team_id, match_date)
    
    # Calculate head-to-head
    h2h = calculate_head_to_head(home_team_id, away_team_id, match_date)
    
    # Calculate form percentage (points per game * 100)
    home_form_pct = (home_form["points"] / (home_form["matches_played"] * 3) * 100) if home_form["matches_played"] > 0 else 0
    away_form_pct = (away_form["points"] / (away_form["matches_played"] * 3) * 100) if away_form["matches_played"] > 0 else 0
    
    return {
        "match_id": match_id,
        "home_last5_points": home_form["points"],
        "home_last5_goal_diff": home_form["goal_diff"],
        "home_form_pct": round(home_form_pct, 2),
        "away_last5_points": away_form["points"],
        "away_last5_goal_diff": away_form["goal_diff"],
        "away_form_pct": round(away_form_pct, 2),
        "h2h_home_wins": h2h["home_wins"],
        "h2h_draws": h2h["draws"],
        "h2h_away_wins": h2h["away_wins"]
    }


def generate_all_features() -> None:
    """
    Generate features for ALL matches in the database.
    
    This processes matches chronologically so each match only uses
    data from BEFORE that match (avoiding data leakage).
    """
    # Get all matches ordered by date
    query = text("""
        SELECT id, match_date, home_team_id, away_team_id
        FROM matches
        ORDER BY match_date ASC
    """)
    
    with engine.connect() as conn:
        matches = conn.execute(query).fetchall()
    
    print(f"Generating features for {len(matches)} matches...")
    
    features_to_insert = []
    
    for idx, (match_id, match_date, home_id, away_id) in enumerate(matches):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(matches)} matches processed")
        
        features = generate_features_for_match(match_id, match_date, home_id, away_id)
        features_to_insert.append(features)
    
    # Insert all features into database
    insert_sql = text("""
        INSERT INTO match_features 
        (match_id, home_last5_points, home_last5_goal_diff, home_form_pct,
         away_last5_points, away_last5_goal_diff, away_form_pct,
         h2h_home_wins, h2h_draws, h2h_away_wins)
        VALUES 
        (:match_id, :home_last5_points, :home_last5_goal_diff, :home_form_pct,
         :away_last5_points, :away_last5_goal_diff, :away_form_pct,
         :h2h_home_wins, :h2h_draws, :h2h_away_wins)
        ON DUPLICATE KEY UPDATE
            home_last5_points = VALUES(home_last5_points),
            home_last5_goal_diff = VALUES(home_last5_goal_diff),
            home_form_pct = VALUES(home_form_pct),
            away_last5_points = VALUES(away_last5_points),
            away_last5_goal_diff = VALUES(away_last5_goal_diff),
            away_form_pct = VALUES(away_form_pct),
            h2h_home_wins = VALUES(h2h_home_wins),
            h2h_draws = VALUES(h2h_draws),
            h2h_away_wins = VALUES(h2h_away_wins)
    """)
    
    with engine.begin() as conn:
        conn.execute(insert_sql, features_to_insert)
    
    print(f"✅ Features generated for {len(features_to_insert)} matches")


if __name__ == "__main__":
    generate_all_features()
