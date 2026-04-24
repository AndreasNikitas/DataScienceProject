"""
Feature Engineering for Football Match Prediction

This script calculates predictive features from historical match data.
These features are used for machine learning WITHOUT data leakage.
"""

import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from sqlalchemy import text

from db import engine
from feature_columns import TRAINING_FEATURE_COLUMNS

_ELO_SNAPSHOT_CACHE: Dict[str, Dict[int, float]] = {}
DEFAULT_FORM_WINDOW = 20


def _to_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _points_from_result(result: str, venue: str) -> int:
    if result == "D":
        return 1
    if venue == "home" and result == "H":
        return 3
    if venue == "away" and result == "A":
        return 3
    return 0


def calculate_team_form(team_id: int, before_date: datetime, num_matches: int = 5) -> Dict:
    before_date = _to_naive(before_date)
    query = text(
        """
        SELECT
            CASE WHEN home_team_id = :team_id THEN 'home' ELSE 'away' END AS venue,
            home_goals,
            away_goals,
            result
        FROM matches
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND match_date < :before_date
          AND result IS NOT NULL
        ORDER BY match_date DESC
        LIMIT :num_matches
        """
    )

    with engine.connect() as conn:
        results = conn.execute(
            query,
            {"team_id": team_id, "before_date": before_date, "num_matches": num_matches},
        ).fetchall()

    if not results:
        return {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "goal_diff": 0,
            "matches_played": 0,
        }

    wins = draws = losses = 0
    goals_scored = goals_conceded = 0

    for venue, home_goals, away_goals, result in results:
        if venue == "home":
            goals_scored += home_goals
            goals_conceded += away_goals
            if result == "H":
                wins += 1
            elif result == "D":
                draws += 1
            else:
                losses += 1
        else:
            goals_scored += away_goals
            goals_conceded += home_goals
            if result == "A":
                wins += 1
            elif result == "D":
                draws += 1
            else:
                losses += 1

    points = wins * 3 + draws
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "goal_diff": goals_scored - goals_conceded,
        "matches_played": len(results),
    }


def calculate_head_to_head(home_team_id: int, away_team_id: int, before_date: datetime) -> Dict:
    before_date = _to_naive(before_date)
    query = text(
        """
        SELECT result
        FROM matches
        WHERE home_team_id = :home_id
          AND away_team_id = :away_id
          AND match_date < :before_date
          AND result IS NOT NULL
        """
    )

    with engine.connect() as conn:
        results = conn.execute(
            query,
            {"home_id": home_team_id, "away_id": away_team_id, "before_date": before_date},
        ).fetchall()

    home_wins = sum(1 for r in results if r[0] == "H")
    draws = sum(1 for r in results if r[0] == "D")
    away_wins = sum(1 for r in results if r[0] == "A")
    return {"home_wins": home_wins, "draws": draws, "away_wins": away_wins}


def calculate_rest_fatigue(team_id: int, before_date: datetime) -> Dict:
    before_date = _to_naive(before_date)
    query = text(
        """
        SELECT
            match_date,
            CASE WHEN home_team_id = :team_id THEN 'home' ELSE 'away' END AS venue
        FROM matches
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND match_date < :before_date
        ORDER BY match_date DESC
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"team_id": team_id, "before_date": before_date}).fetchall()

    if not rows:
        return {
            "days_since_last_match": 14.0,
            "matches_last7": 0,
            "matches_last14": 0,
            "last_venue": "home",
        }

    last_match_date, last_venue = rows[0]
    days_since = (before_date - last_match_date).total_seconds() / 86400
    threshold_7 = before_date - timedelta(days=7)
    threshold_14 = before_date - timedelta(days=14)
    matches_last7 = sum(1 for r in rows if r[0] >= threshold_7)
    matches_last14 = sum(1 for r in rows if r[0] >= threshold_14)

    return {
        "days_since_last_match": max(0.0, days_since),
        "matches_last7": matches_last7,
        "matches_last14": matches_last14,
        "last_venue": last_venue,
    }


def calculate_venue_strength(team_id: int, before_date: datetime, venue: str, num_matches: int = 10) -> Dict:
    before_date = _to_naive(before_date)
    if venue == "home":
        where_clause = "home_team_id = :team_id"
        result_as_win = "H"
    else:
        where_clause = "away_team_id = :team_id"
        result_as_win = "A"

    query = text(
        f"""
        SELECT home_goals, away_goals, result
        FROM matches
        WHERE {where_clause}
          AND match_date < :before_date
          AND result IS NOT NULL
        ORDER BY match_date DESC
        LIMIT :num_matches
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"team_id": team_id, "before_date": before_date, "num_matches": num_matches},
        ).fetchall()

    if not rows:
        return {"ppg": 1.0, "goal_diff_avg": 0.0}

    points = 0
    goal_diff = 0
    for home_goals, away_goals, result in rows:
        if result == "D":
            points += 1
        elif result == result_as_win:
            points += 3

        if venue == "home":
            goal_diff += home_goals - away_goals
        else:
            goal_diff += away_goals - home_goals

    matches = len(rows)
    return {"ppg": points / matches, "goal_diff_avg": goal_diff / matches}


def calculate_overall_strength(team_id: int, before_date: datetime, num_matches: int = 30) -> Dict:
    form = calculate_team_form(team_id, before_date, num_matches=num_matches)
    matches = max(form["matches_played"], 1)
    return {
        "ppg": form["points"] / matches,
        "goal_diff_avg": form["goal_diff"] / matches,
    }


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10 ** ((rating_b - rating_a) / 400.0)))


def calculate_elo_before_match(home_team_id: int, away_team_id: int, before_date: datetime, k_factor: float = 24.0) -> Dict:
    before_date = _to_naive(before_date)
    cache_key = before_date.isoformat()
    elo = _ELO_SNAPSHOT_CACHE.get(cache_key)
    if elo is None:
        query = text(
            """
            SELECT home_team_id, away_team_id, result
            FROM matches
            WHERE result IS NOT NULL
              AND match_date < :before_date
            ORDER BY match_date ASC, id ASC
            """
        )
        with engine.connect() as conn:
            rows = conn.execute(query, {"before_date": before_date}).fetchall()

        elo = {}
        for home_id, away_id, result in rows:
            home_rating = elo.get(home_id, 1500.0)
            away_rating = elo.get(away_id, 1500.0)
            expected_home = _elo_expected(home_rating, away_rating)
            expected_away = _elo_expected(away_rating, home_rating)

            if result == "H":
                actual_home, actual_away = 1.0, 0.0
            elif result == "A":
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home = actual_away = 0.5

            elo[home_id] = home_rating + k_factor * (actual_home - expected_home)
            elo[away_id] = away_rating + k_factor * (actual_away - expected_away)
        _ELO_SNAPSHOT_CACHE[cache_key] = elo

    home_elo = elo.get(home_team_id, 1500.0)
    away_elo = elo.get(away_team_id, 1500.0)
    return {"home_elo": home_elo, "away_elo": away_elo, "elo_diff": home_elo - away_elo}


def calculate_travel_penalty(rest: Dict, is_away_team: bool) -> float:
    penalty = 0.0
    if rest["days_since_last_match"] < 4:
        penalty += 0.10 if is_away_team else 0.05
    if rest["matches_last14"] >= 5:
        penalty += 0.10 if is_away_team else 0.05
    if is_away_team and rest["last_venue"] == "away":
        penalty += 0.10
    return min(0.30, penalty)


def generate_features_for_match(
    match_id: int,
    match_date: datetime,
    home_team_id: int,
    away_team_id: int,
    form_window: int = 5,
) -> Dict:
    match_date = _to_naive(match_date)

    home_form = calculate_team_form(home_team_id, match_date, num_matches=form_window)
    away_form = calculate_team_form(away_team_id, match_date, num_matches=form_window)
    h2h = calculate_head_to_head(home_team_id, away_team_id, match_date)
    home_rest = calculate_rest_fatigue(home_team_id, match_date)
    away_rest = calculate_rest_fatigue(away_team_id, match_date)
    home_home_strength = calculate_venue_strength(home_team_id, match_date, venue="home", num_matches=10)
    away_away_strength = calculate_venue_strength(away_team_id, match_date, venue="away", num_matches=10)
    home_overall_strength = calculate_overall_strength(home_team_id, match_date, num_matches=30)
    away_overall_strength = calculate_overall_strength(away_team_id, match_date, num_matches=30)
    elo = calculate_elo_before_match(home_team_id, away_team_id, match_date)

    home_form_pct = (
        (home_form["points"] / (home_form["matches_played"] * 3) * 100)
        if home_form["matches_played"] > 0
        else 0.0
    )
    away_form_pct = (
        (away_form["points"] / (away_form["matches_played"] * 3) * 100)
        if away_form["matches_played"] > 0
        else 0.0
    )

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
        "h2h_away_wins": h2h["away_wins"],
        "home_days_since_last_match": round(home_rest["days_since_last_match"], 2),
        "away_days_since_last_match": round(away_rest["days_since_last_match"], 2),
        "home_matches_last7": home_rest["matches_last7"],
        "away_matches_last7": away_rest["matches_last7"],
        "home_matches_last14": home_rest["matches_last14"],
        "away_matches_last14": away_rest["matches_last14"],
        "home_travel_penalty": round(calculate_travel_penalty(home_rest, is_away_team=False), 3),
        "away_travel_penalty": round(calculate_travel_penalty(away_rest, is_away_team=True), 3),
        "home_home_ppg_last10": round(home_home_strength["ppg"], 3),
        "away_away_ppg_last10": round(away_away_strength["ppg"], 3),
        "home_home_goal_diff_last10": round(home_home_strength["goal_diff_avg"], 3),
        "away_away_goal_diff_last10": round(away_away_strength["goal_diff_avg"], 3),
        "home_overall_ppg_last30": round(home_overall_strength["ppg"], 3),
        "away_overall_ppg_last30": round(away_overall_strength["ppg"], 3),
        "home_overall_goal_diff_last30": round(home_overall_strength["goal_diff_avg"], 3),
        "away_overall_goal_diff_last30": round(away_overall_strength["goal_diff_avg"], 3),
        "home_elo": round(elo["home_elo"], 3),
        "away_elo": round(elo["away_elo"], 3),
        "elo_diff": round(elo["elo_diff"], 3),
    }


def ensure_match_features_columns() -> None:
    required_columns = {
        "home_days_since_last_match": "DECIMAL(6,2) DEFAULT 14.0",
        "away_days_since_last_match": "DECIMAL(6,2) DEFAULT 14.0",
        "home_matches_last7": "INT DEFAULT 0",
        "away_matches_last7": "INT DEFAULT 0",
        "home_matches_last14": "INT DEFAULT 0",
        "away_matches_last14": "INT DEFAULT 0",
        "home_travel_penalty": "DECIMAL(5,3) DEFAULT 0.000",
        "away_travel_penalty": "DECIMAL(5,3) DEFAULT 0.000",
        "home_home_ppg_last10": "DECIMAL(6,3) DEFAULT 0.000",
        "away_away_ppg_last10": "DECIMAL(6,3) DEFAULT 0.000",
        "home_home_goal_diff_last10": "DECIMAL(6,3) DEFAULT 0.000",
        "away_away_goal_diff_last10": "DECIMAL(6,3) DEFAULT 0.000",
        "home_overall_ppg_last30": "DECIMAL(6,3) DEFAULT 0.000",
        "away_overall_ppg_last30": "DECIMAL(6,3) DEFAULT 0.000",
        "home_overall_goal_diff_last30": "DECIMAL(6,3) DEFAULT 0.000",
        "away_overall_goal_diff_last30": "DECIMAL(6,3) DEFAULT 0.000",
        "home_elo": "DECIMAL(8,3) DEFAULT 1500.0",
        "away_elo": "DECIMAL(8,3) DEFAULT 1500.0",
        "elo_diff": "DECIMAL(8,3) DEFAULT 0.0",
    }

    query = text(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'match_features'
        """
    )

    with engine.begin() as conn:
        existing = {row[0] for row in conn.execute(query).fetchall()}
        for column, definition in required_columns.items():
            if column in existing:
                continue
            conn.execute(text(f"ALTER TABLE match_features ADD COLUMN {column} {definition}"))


def generate_all_features(form_window: int = DEFAULT_FORM_WINDOW) -> None:
    ensure_match_features_columns()

    query = text(
        """
        SELECT id, match_date, home_team_id, away_team_id
        FROM matches
        ORDER BY match_date ASC
        """
    )
    with engine.connect() as conn:
        matches = conn.execute(query).fetchall()

    print(f"Generating features for {len(matches)} matches...")
    features_to_insert: List[Dict] = []
    for idx, (match_id, match_date, home_id, away_id) in enumerate(matches):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(matches)} matches processed")
        features_to_insert.append(
            generate_features_for_match(
                match_id=match_id,
                match_date=match_date,
                home_team_id=home_id,
                away_team_id=away_id,
                form_window=form_window,
            )
        )

    insert_columns = ["match_id"] + TRAINING_FEATURE_COLUMNS
    column_sql = ", ".join(insert_columns)
    value_sql = ", ".join(f":{col}" for col in insert_columns)
    update_sql = ", ".join(f"{col} = VALUES({col})" for col in TRAINING_FEATURE_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO match_features ({column_sql})
        VALUES ({value_sql})
        ON DUPLICATE KEY UPDATE {update_sql}
        """
    )

    with engine.begin() as conn:
        conn.execute(insert_sql, features_to_insert)

    print(f"✅ Features generated for {len(features_to_insert)} matches using form window {form_window}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training features for all matches.")
    parser.add_argument(
        "--form-window",
        type=int,
        default=DEFAULT_FORM_WINDOW,
        help="Recent-form window to use when generating match_features",
    )
    args = parser.parse_args()
    generate_all_features(form_window=args.form_window)
