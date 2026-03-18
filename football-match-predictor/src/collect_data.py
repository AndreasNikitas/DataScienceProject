from typing import Dict, List

from sqlalchemy import text

from db import engine


def fetch_matches() -> List[Dict]:
    """Placeholder function for fetching match data from an external API."""
    # TODO: Add API endpoint and authentication details.
    # TODO: Make the API request.
    # TODO: Parse response into match dictionaries.
    return []




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
