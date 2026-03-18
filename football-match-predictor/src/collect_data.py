from datetime import datetime
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


def fetch_matches(league: str = "eng.1") -> List[Dict]:
    """Fetch matches from the ESPN scoreboard API for the given league slug.

    Args:
        league: ESPN league slug, e.g. "eng.1" (Premier League),
                "esp.1" (La Liga), "ger.1" (Bundesliga).

    Returns:
        List of match dicts ready for ``insert_matches``.
    """
    url = f"{_ESPN_BASE}/{league}/scoreboard"
    response = requests.get(url, timeout=10)
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


def insert_matches(matches: List[Dict]) -> None:
    """Example insert structure for saving fetched matches."""
    if not matches:
        print("No matches to insert.")
        return

    insert_sql = text(
        """
        INSERT INTO matches (match_date, home_team_id, away_team_id, home_goals, away_goals, result)
        VALUES (:match_date, :home_team_id, :away_team_id, :home_goals, :away_goals, :result)
        """
    )

    with engine.begin() as connection:
        for match in matches:
            connection.execute(insert_sql, match)

    print(f"Inserted {len(matches)} matches.")


def main() -> None:
    matches = fetch_matches()
    # TODO: Apply data cleanup/transformation before insert.
    insert_matches(matches)


if __name__ == "__main__":
    main()
