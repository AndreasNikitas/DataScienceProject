from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from sqlalchemy import text

from db import engine

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def _get_or_create_team(name: str) -> int:
    """Return the DB id for a team, inserting it if it doesn't exist yet."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM teams WHERE name = :name"), {"name": name}
        ).fetchone()
        if row:
            return row[0]
        result = conn.execute(
            text("INSERT INTO teams (name) VALUES (:name)"), {"name": name}
        )
        return result.lastrowid


def fetch_matches(league: str = "eng.1", dates: Optional[str] = None) -> List[Dict]:
    """Fetch matches from the ESPN scoreboard API for the given league slug.

    Args:
        league: ESPN league slug, e.g. "eng.1" (Premier League),
                "esp.1" (La Liga), "ger.1" (Bundesliga).
        dates: Optional date range in format "YYYYMMDD-YYYYMMDD" (e.g. "20240101-20240131")

    Returns:
        List of match dicts ready for ``insert_matches``.
    """
    url = f"{_ESPN_BASE}/{league}/scoreboard"
    params = {"dates": dates} if dates else {}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    matches: List[Dict] = []
    for event in response.json().get("events", []):
        competition = event["competitions"][0]
        competitors = competition["competitors"]

        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue

        is_finished = event["status"]["type"]["state"] == "post"
        home_goals: Optional[int] = int(home["score"]) if is_finished else None
        away_goals: Optional[int] = int(away["score"]) if is_finished else None

        result: Optional[str] = None
        if home_goals is not None and away_goals is not None:
            if home_goals > away_goals:
                result = "H"
            elif away_goals > home_goals:
                result = "A"
            else:
                result = "D"

        match_date = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))

        matches.append({
            "match_date": match_date,
            "home_team_id": _get_or_create_team(home["team"]["displayName"]),
            "away_team_id": _get_or_create_team(away["team"]["displayName"]),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
        })

    return matches


def validate_match(match: Dict) -> bool:
    """Validate match data before insertion.
    
    Checks:
    - Goals are non-negative (if present)
    - Home and away teams are different
    - Date is valid
    """
    # Check goals are non-negative
    if match["home_goals"] is not None and match["home_goals"] < 0:
        print(f"⚠️  Invalid home_goals: {match['home_goals']}")
        return False
    if match["away_goals"] is not None and match["away_goals"] < 0:
        print(f"⚠️  Invalid away_goals: {match['away_goals']}")
        return False
    
    # Check teams are different
    if match["home_team_id"] == match["away_team_id"]:
        print(f"⚠️  Same team playing itself: {match['home_team_id']}")
        return False
    
    return True


def match_exists(connection, match: Dict) -> bool:
    """Check if a match already exists in the database."""
    check_sql = text(
        """
        SELECT COUNT(*) FROM matches 
        WHERE match_date = :match_date 
        AND home_team_id = :home_team_id 
        AND away_team_id = :away_team_id
        """
    )
    result = connection.execute(check_sql, match).fetchone()
    return result[0] > 0


def insert_matches(matches: List[Dict]) -> None:
    """Insert matches with validation and duplicate checking."""
    if not matches:
        print("No matches to insert.")
        return

    insert_sql = text(
        """
        INSERT INTO matches (match_date, home_team_id, away_team_id, home_goals, away_goals, result)
        VALUES (:match_date, :home_team_id, :away_team_id, :home_goals, :away_goals, :result)
        """
    )

    inserted = 0
    skipped = 0
    
    with engine.begin() as connection:
        for match in matches:
            # Validate match data
            if not validate_match(match):
                skipped += 1
                continue
            
            # Check for duplicates
            if match_exists(connection, match):
                skipped += 1
                continue
            
            # Insert valid, non-duplicate match
            connection.execute(insert_sql, match)
            inserted += 1

    print(f"Inserted {inserted} matches, skipped {skipped} duplicates/invalid.")


def fetch_historical_data(league: str = "eng.1", months_back: int = 12) -> None:
    """Fetch historical match data for the past N months.
    
    Args:
        league: ESPN league slug
        months_back: Number of months of historical data to fetch
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months_back)
    
    print(f"Fetching matches from {start_date.date()} to {end_date.date()}...")
    
    # Fetch data in monthly chunks
    current = start_date
    total_matches = 0
    
    while current < end_date:
        chunk_end = min(current + timedelta(days=30), end_date)
        date_range = f"{current.strftime('%Y%m%d')}-{chunk_end.strftime('%Y%m%d')}"
        
        print(f"  Fetching: {date_range}")
        try:
            matches = fetch_matches(league, date_range)
            insert_matches(matches)
            total_matches += len(matches)
        except Exception as e:
            print(f"  Error fetching {date_range}: {e}")
        
        current = chunk_end + timedelta(days=1)
    
    print(f"\nTotal matches fetched: {total_matches}")


def main() -> None:
    # Fetch historical data (last 24 months for better predictions)
    fetch_historical_data(league="eng.1", months_back=24)


if __name__ == "__main__":
    main()
