import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text

from db import engine

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
_VALID_RESULTS = {"H", "D", "A"}
_MAX_REASONABLE_GOALS = 20
_STALE_UPCOMING_GRACE_HOURS = 6


def _to_utc_naive(value: datetime) -> datetime:
    """Normalize datetime to UTC-naive for consistent comparisons/writes."""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _is_stale_upcoming(match_date: datetime, grace_hours: int = _STALE_UPCOMING_GRACE_HOURS) -> bool:
    """True when fixture date is in the past beyond grace period and still unresolved."""
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    normalized_match_date = _to_utc_naive(match_date)
    return normalized_match_date < (now_utc - timedelta(hours=grace_hours))


def normalize_team_name(name: Optional[str]) -> Optional[str]:
    """Normalize team names by trimming and collapsing repeated spaces."""
    if not isinstance(name, str):
        return None
    normalized = " ".join(name.split())
    return normalized or None


def _safe_int(value: Any) -> Optional[int]:
    """Parse an integer safely, returning None for empty/invalid values."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _derive_result(home_goals: int, away_goals: int) -> str:
    """Derive H/D/A result from home and away goals."""
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def _parse_match_date(raw_date: Any) -> Optional[datetime]:
    """Parse ESPN ISO date safely."""
    if not isinstance(raw_date, str) or not raw_date.strip():
        return None
    try:
        return datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError:
        return None


def _get_or_create_team(name: str) -> int:
    """Return the DB id for a team, inserting it if it doesn't exist yet."""
    cleaned_name = normalize_team_name(name)
    if cleaned_name is None:
        raise ValueError("Team name is empty after normalization")

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM teams WHERE LOWER(name) = LOWER(:name)"), {"name": cleaned_name}
        ).fetchone()
        if row:
            return row[0]
        result = conn.execute(
            text("INSERT INTO teams (name) VALUES (:name)"), {"name": cleaned_name}
        )
        return result.lastrowid


def clean_match(event: Dict[str, Any]) -> Optional[Dict]:
    """Clean and normalize raw ESPN event payload into match dict.

    Returns None when required fields are missing/invalid.
    """
    competitions = event.get("competitions")
    if not competitions:
        return None

    competition = competitions[0]
    competitors = competition.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    home_name = normalize_team_name(home.get("team", {}).get("displayName"))
    away_name = normalize_team_name(away.get("team", {}).get("displayName"))
    if home_name is None or away_name is None:
        return None

    match_date = _parse_match_date(event.get("date"))
    if match_date is None:
        return None

    is_finished = event.get("status", {}).get("type", {}).get("state") == "post"
    home_goals: Optional[int] = _safe_int(home.get("score")) if is_finished else None
    away_goals: Optional[int] = _safe_int(away.get("score")) if is_finished else None

    if is_finished and (home_goals is None or away_goals is None):
        return None

    result: Optional[str] = (
        _derive_result(home_goals, away_goals)
        if home_goals is not None and away_goals is not None
        else None
    )

    # Do not keep unresolved fixtures that are already in the past.
    if result is None and _is_stale_upcoming(match_date):
        return None

    return {
        "match_date": _to_utc_naive(match_date),
        "home_team_id": _get_or_create_team(home_name),
        "away_team_id": _get_or_create_team(away_name),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
    }


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
    skipped_unclean = 0
    for event in response.json().get("events", []):
        cleaned_match = clean_match(event)
        if cleaned_match is None:
            skipped_unclean += 1
            continue
        matches.append(cleaned_match)

    if skipped_unclean:
        print(f"Skipped {skipped_unclean} events with missing/invalid fields.")

    return matches


def validate_match(match: Dict) -> bool:
    """Validate match data before insertion.
    
    Checks:
    - match_date exists and is datetime
    - team ids are positive and different
    - goals are integers in a realistic range (if present)
    - score/result are mutually consistent
    """
    if not isinstance(match.get("match_date"), datetime):
        print(f"⚠️  Invalid match_date: {match.get('match_date')}")
        return False

    home_team_id = match.get("home_team_id")
    away_team_id = match.get("away_team_id")
    if not isinstance(home_team_id, int) or home_team_id <= 0:
        print(f"⚠️  Invalid home_team_id: {home_team_id}")
        return False
    if not isinstance(away_team_id, int) or away_team_id <= 0:
        print(f"⚠️  Invalid away_team_id: {away_team_id}")
        return False

    # Check teams are different
    if home_team_id == away_team_id:
        print(f"⚠️  Same team playing itself: {home_team_id}")
        return False

    home_goals = match.get("home_goals")
    away_goals = match.get("away_goals")
    result = match.get("result")

    if home_goals is not None and not isinstance(home_goals, int):
        print(f"⚠️  home_goals must be int/None: {home_goals}")
        return False
    if away_goals is not None and not isinstance(away_goals, int):
        print(f"⚠️  away_goals must be int/None: {away_goals}")
        return False

    # Check goals are non-negative and realistic
    if home_goals is not None and (home_goals < 0 or home_goals > _MAX_REASONABLE_GOALS):
        print(f"⚠️  Invalid home_goals: {match['home_goals']}")
        return False
    if away_goals is not None and (away_goals < 0 or away_goals > _MAX_REASONABLE_GOALS):
        print(f"⚠️  Invalid away_goals: {match['away_goals']}")
        return False

    if result is not None and result not in _VALID_RESULTS:
        print(f"⚠️  Invalid result value: {result}")
        return False

    # Require goals and result to either all exist (finished) or all be null (upcoming)
    if (home_goals is None) != (away_goals is None):
        print("⚠️  Partial score found (one goal is missing).")
        return False

    if home_goals is None and away_goals is None:
        if result is not None:
            print("⚠️  Upcoming match has a result.")
            return False
        if _is_stale_upcoming(match["match_date"]):
            print(f"⚠️  Stale upcoming fixture in the past: {match['match_date']}")
            return False
        return True

    # Finished match: ensure result matches the scoreline
    expected_result = _derive_result(home_goals, away_goals)
    if result != expected_result:
        print(
            f"⚠️  Result/score mismatch. result={result}, expected={expected_result} "
            f"from score {home_goals}-{away_goals}"
        )
        return False

    return True


def insert_matches(matches: List[Dict]) -> None:
    """Insert matches and update existing fixtures when final score appears."""
    if not matches:
        print("No matches to insert.")
        return

    insert_sql = text(
        """
        INSERT INTO matches (match_date, home_team_id, away_team_id, home_goals, away_goals, result)
        VALUES (:match_date, :home_team_id, :away_team_id, :home_goals, :away_goals, :result)
        ON DUPLICATE KEY UPDATE
            home_goals = COALESCE(VALUES(home_goals), home_goals),
            away_goals = COALESCE(VALUES(away_goals), away_goals),
            result = COALESCE(VALUES(result), result)
        """
    )

    written = 0
    skipped = 0
    
    with engine.begin() as connection:
        for match in matches:
            # Validate match data
            if not validate_match(match):
                skipped += 1
                continue
            connection.execute(insert_sql, match)
            written += 1

    print(f"Upserted {written} matches, skipped {skipped} invalid.")


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


def fetch_upcoming_data(league: str = "eng.1", days_ahead: int = 14) -> None:
    """Fetch upcoming fixtures for the next N days."""
    if days_ahead <= 0:
        return

    start_date = datetime.now()
    end_date = start_date + timedelta(days=days_ahead)
    date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    print(f"Fetching upcoming fixtures for {date_range}...")
    matches = fetch_matches(league=league, dates=date_range)
    insert_matches(matches)
    print(f"Fetched {len(matches)} upcoming/ongoing fixtures in lookahead window.")


def remove_stale_upcoming_matches(grace_hours: int = _STALE_UPCOMING_GRACE_HOURS) -> None:
    """Remove unresolved past fixtures and dependent pending artifacts."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=grace_hours)

    with engine.begin() as conn:
        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS stale_matches"))
        conn.execute(
            text(
                """
                CREATE TEMPORARY TABLE stale_matches AS
                SELECT id
                FROM matches
                WHERE result IS NULL
                  AND match_date < :cutoff
                  AND id NOT IN (SELECT match_id FROM match_predictions)
                """
            ),
            {"cutoff": cutoff},
        )

        stale_count = conn.execute(text("SELECT COUNT(*) FROM stale_matches")).scalar() or 0
        if stale_count == 0:
            conn.execute(text("DROP TEMPORARY TABLE IF EXISTS stale_matches"))
            return

        conn.execute(
            text(
                """
                DELETE mp
                FROM match_predictions mp
                JOIN stale_matches sm ON sm.id = mp.match_id
                """
            )
        )
        conn.execute(
            text(
                """
                DELETE mf
                FROM match_features mf
                JOIN stale_matches sm ON sm.id = mf.match_id
                """
            )
        )
        conn.execute(
            text(
                """
                DELETE m
                FROM matches m
                JOIN stale_matches sm ON sm.id = m.id
                """
            )
        )
        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS stale_matches"))

    print(f"Removed {stale_count} stale unresolved past fixture(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect football matches from ESPN API.")
    parser.add_argument("--league", default="eng.1", help="ESPN league slug, e.g. eng.1, esp.1")
    parser.add_argument(
        "--months-back",
        type=int,
        default=24,
        help="How many months of historical matches to fetch",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=14,
        help="How many days of upcoming fixtures to fetch",
    )
    args = parser.parse_args()

    remove_stale_upcoming_matches()
    fetch_historical_data(league=args.league, months_back=args.months_back)
    fetch_upcoming_data(league=args.league, days_ahead=args.days_ahead)


if __name__ == "__main__":
    main()
