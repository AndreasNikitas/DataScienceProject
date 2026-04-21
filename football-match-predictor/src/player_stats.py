from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd
import requests


_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def fetch_league_teams(league: str) -> List[Dict[str, Any]]:
    response = requests.get(f"{_ESPN_BASE}/{league}/teams", timeout=15)
    response.raise_for_status()
    payload = response.json()
    return payload["sports"][0]["leagues"][0].get("teams", [])


def resolve_team_id(league: str, team_name: str) -> int:
    normalized = team_name.strip().lower()
    if not normalized:
        raise ValueError("Team name cannot be empty.")

    teams = fetch_league_teams(league)
    for entry in teams:
        team = entry.get("team", {})
        aliases = {
            str(team.get("displayName", "")).lower(),
            str(team.get("shortDisplayName", "")).lower(),
            str(team.get("name", "")).lower(),
            str(team.get("abbreviation", "")).lower(),
        }
        if normalized in aliases:
            return int(team["id"])

    raise ValueError(f"Team '{team_name}' not found in league '{league}'.")


def fetch_team_roster(league: str, team_id: int) -> Dict[str, Any]:
    response = requests.get(f"{_ESPN_BASE}/{league}/teams/{team_id}/roster", timeout=15)
    response.raise_for_status()
    return response.json()


def _extract_stat(stat_categories: List[Dict[str, Any]], stat_name: str) -> float:
    for category in stat_categories:
        for stat in category.get("stats", []):
            if stat.get("name") == stat_name:
                value = stat.get("value")
                return float(value) if value is not None else 0.0
    return 0.0


def roster_to_dataframes(roster_payload: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    athletes = roster_payload.get("athletes", [])
    player_rows: List[Dict[str, Any]] = []
    unavailable_rows: List[Dict[str, Any]] = []

    for athlete in athletes:
        splits = athlete.get("statistics", {}).get("splits", {})
        categories = splits.get("categories", [])
        injuries = athlete.get("injuries", [])

        row = {
            "player": athlete.get("displayName"),
            "position": athlete.get("position", {}).get("abbreviation"),
            "status": athlete.get("status", {}).get("name", "Unknown"),
            "appearances": _extract_stat(categories, "appearances"),
            "goals": _extract_stat(categories, "totalGoals"),
            "assists": _extract_stat(categories, "goalAssists"),
            "shots_on_target": _extract_stat(categories, "shotsOnTarget"),
            "yellow_cards": _extract_stat(categories, "yellowCards"),
            "red_cards": _extract_stat(categories, "redCards"),
            "injury_count": len(injuries),
        }
        player_rows.append(row)

        for injury in injuries:
            unavailable_rows.append(
                {
                    "player": athlete.get("displayName"),
                    "position": athlete.get("position", {}).get("abbreviation"),
                    "injury": injury.get("type", {}).get("name", "Unknown"),
                    "status": injury.get("status", "Unknown"),
                    "detail": injury.get("details", "n/a"),
                }
            )

    players_df = pd.DataFrame(player_rows)
    unavailable_df = pd.DataFrame(unavailable_rows)
    return {"players": players_df, "unavailable": unavailable_df}


def get_team_player_report(league: str, team_name: str) -> Dict[str, pd.DataFrame]:
    team_id = resolve_team_id(league, team_name)
    roster = fetch_team_roster(league, team_id)
    return roster_to_dataframes(roster)


def compute_availability_impact(players_df: pd.DataFrame, unavailable_df: pd.DataFrame) -> Dict[str, Any]:
    if players_df.empty:
        return {
            "injured_players": 0,
            "top_scorer_absences": 0,
            "missing_goal_share": 0.0,
            "availability_penalty": 0.0,
            "key_absences": [],
        }

    goals_series = pd.to_numeric(players_df["goals"], errors="coerce").fillna(0.0)
    total_goals = float(goals_series.sum())

    unavailable_names = set(unavailable_df["player"].tolist()) if not unavailable_df.empty else set()
    injured_players = len(unavailable_names)

    missing_goals = 0.0
    key_absences: List[str] = []
    top_scorer_absences = 0

    if unavailable_names:
        ranked = players_df.copy()
        ranked["goals"] = goals_series
        ranked = ranked.sort_values(["goals", "assists"], ascending=False)
        top_players = ranked.head(5)

        for _, row in ranked.iterrows():
            if row["player"] in unavailable_names:
                missing_goals += float(row["goals"])

        for _, row in top_players.iterrows():
            if row["player"] in unavailable_names and float(row["goals"]) > 0:
                top_scorer_absences += 1
                key_absences.append(str(row["player"]))

    missing_goal_share = (missing_goals / total_goals) if total_goals > 0 else 0.0
    availability_penalty = min(
        0.35,
        (missing_goal_share * 0.60) + (min(injured_players, 5) * 0.02) + (top_scorer_absences * 0.05),
    )

    return {
        "injured_players": injured_players,
        "top_scorer_absences": top_scorer_absences,
        "missing_goal_share": float(missing_goal_share),
        "availability_penalty": float(availability_penalty),
        "key_absences": key_absences,
    }


def get_team_availability_impact(league: str, team_name: str) -> Dict[str, Any]:
    try:
        report = get_team_player_report(league, team_name)
        impact = compute_availability_impact(report["players"], report["unavailable"])
        impact["available"] = True
        return impact
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {
            "injured_players": 0,
            "top_scorer_absences": 0,
            "missing_goal_share": 0.0,
            "availability_penalty": 0.0,
            "key_absences": [],
            "available": False,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch player statistics and injury status.")
    parser.add_argument("--league", default="eng.1", help="ESPN league slug, e.g. eng.1, esp.1")
    parser.add_argument("--team", required=True, help="Team display name, e.g. Arsenal")
    args = parser.parse_args()

    report = get_team_player_report(args.league, args.team)
    players = report["players"].sort_values(["goals", "assists"], ascending=False)
    unavailable = report["unavailable"]

    print(f"\nTop player stats for {args.team} ({args.league}):")
    print(players.head(15).to_string(index=False))

    if unavailable.empty:
        print("\nNo injuries/unavailable players reported in roster feed.")
    else:
        print("\nUnavailable / injury report:")
        print(unavailable.to_string(index=False))

    impact = compute_availability_impact(players, unavailable)
    print(
        "\nAvailability impact:"
        f" injured_players={impact['injured_players']},"
        f" top_scorer_absences={impact['top_scorer_absences']},"
        f" missing_goal_share={impact['missing_goal_share']:.1%},"
        f" penalty={impact['availability_penalty']:.1%}"
    )


if __name__ == "__main__":
    main()
