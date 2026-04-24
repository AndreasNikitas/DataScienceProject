"""
Football Match Predictor Dashboard
"""

import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db import engine
from player_stats import get_team_player_report

st.set_page_config(page_title="Football Match Predictor", page_icon="🏆", layout="wide")
st.title("🏆 Football Match Predictor Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🎯 Dashboard",
        "🔮 Predictions",
        "✅ Accuracy",
        "📈 Statistics",
        "👤 Player Stats",
        "🏥 Data Quality",
    ],
)

if page == "🎯 Dashboard":
    st.header("Welcome to Football Match Predictor")

    col1, col2, col3 = st.columns(3)

    with col1:
        result = pd.read_sql("SELECT COUNT(*) as count FROM matches", engine)
        st.metric("Total Matches", result["count"].values[0] if not result.empty else 0)

    with col2:
        result = pd.read_sql("SELECT COUNT(*) as count FROM teams", engine)
        st.metric("Teams", result["count"].values[0] if not result.empty else 0)

    with col3:
        result = pd.read_sql("SELECT COUNT(*) as count FROM matches WHERE result IS NOT NULL", engine)
        st.metric("Finished Matches", result["count"].values[0] if not result.empty else 0)

    st.markdown("---")
    st.subheader("📋 System Overview")
    st.markdown(
        """
        This dashboard uses **two ML models** for predictions:
        - **Random Forest**: Ensemble learning
        - **Logistic Regression**: Linear classification

        Features analyzed:
        - Recent team form
        - Goal trends
        - Head-to-head history
        """
    )

elif page == "🔮 Predictions":
    st.header("Upcoming Match Predictions")
    stale_df = pd.read_sql(
        """
        SELECT
            COUNT(*) AS stale_matches
        FROM matches
        WHERE result IS NULL
          AND match_date < UTC_TIMESTAMP()
        """,
        engine,
    )
    stale_matches = int(stale_df.iloc[0]["stale_matches"]) if not stale_df.empty else 0
    if stale_matches > 0:
        st.warning(
            f"{stale_matches} match(es) are in the past but still have no final result. "
            "Run data collection to refresh completed scores."
        )

    upcoming_df = pd.read_sql(
        """
        SELECT
            m.match_date,
            t1.name AS home_team,
            t2.name AS away_team,
            mp.predicted_home_goals,
            mp.predicted_away_goals,
            mp.predicted_result,
            mp.rf_prediction,
            mp.rf_confidence,
            mp.lr_prediction,
            mp.lr_confidence
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        LEFT JOIN (
            SELECT mp1.*
            FROM match_predictions mp1
            JOIN (
                SELECT match_id, MAX(created_at) AS created_at
                FROM match_predictions
                GROUP BY match_id
            ) latest
              ON latest.match_id = mp1.match_id
             AND latest.created_at = mp1.created_at
        ) mp ON mp.match_id = m.id
        WHERE m.result IS NULL
          AND m.match_date >= UTC_TIMESTAMP()
        ORDER BY m.match_date ASC
        LIMIT 100
        """,
        engine,
    )

    if upcoming_df.empty:
        st.info("No upcoming matches with stored predictions yet. Run: python src/predict_upcoming.py")
    else:
        outcome_map = {"H": "Home", "D": "Draw", "A": "Away"}
        upcoming_df["predicted_winner"] = upcoming_df["predicted_result"].map(outcome_map)
        upcoming_df["rf_outcome"] = upcoming_df["rf_prediction"].map(outcome_map)
        upcoming_df["lr_outcome"] = upcoming_df["lr_prediction"].map(outcome_map)
        for col in ("rf_confidence", "lr_confidence"):
            upcoming_df[col] = (upcoming_df[col] * 100).round(1).astype(str) + "%"

        st.dataframe(
            upcoming_df[
                [
                    "match_date",
                    "home_team",
                    "away_team",
                    "predicted_home_goals",
                    "predicted_away_goals",
                    "predicted_winner",
                    "rf_outcome",
                    "rf_confidence",
                    "lr_outcome",
                    "lr_confidence",
                ]
            ],
            use_container_width=True,
        )
        st.caption(
            "RF/LR confidence = model probability for its own H/D/A outcome prediction, "
            "not confidence on exact score."
        )

elif page == "✅ Accuracy":
    st.header("Model Accuracy & Evaluation")
 

    summary_df = pd.read_sql(
        """
        SELECT
            COUNT(*) AS resolved_matches,
            AVG(outcome_correct) * 100 AS outcome_accuracy,
            AVG(score_exact) * 100 AS exact_score_accuracy,
            AVG(CASE WHEN rf_prediction = actual_result THEN 1 ELSE 0 END) * 100 AS rf_accuracy,
            AVG(CASE WHEN lr_prediction = actual_result THEN 1 ELSE 0 END) * 100 AS lr_accuracy,
            AVG(CASE
                WHEN consensus_prediction IS NOT NULL AND consensus_prediction = actual_result THEN 1
                ELSE 0
            END) * 100 AS consensus_accuracy
        FROM match_predictions
        WHERE status = 'resolved'
        """,
        engine,
    )
    summary = summary_df.iloc[0] if not summary_df.empty else None
    resolved_matches = int(summary["resolved_matches"] or 0) if summary is not None else 0

    if resolved_matches == 0:
        st.info("No resolved predictions yet. After matches finish, run prediction pipeline again.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Resolved matches", resolved_matches)
        with c2:
            st.metric("Outcome accuracy", f"{float(summary['outcome_accuracy']):.1f}%")
        with c3:
            st.metric("Exact score accuracy", f"{float(summary['exact_score_accuracy']):.1f}%")

        model_accuracy = pd.DataFrame(
            {
                "model": ["Random Forest", "Logistic Regression", "Consensus"],
                "accuracy_pct": [
                    float(summary["rf_accuracy"] or 0.0),
                    float(summary["lr_accuracy"] or 0.0),
                    float(summary["consensus_accuracy"] or 0.0),
                ],
            }
        )
        st.subheader("Outcome accuracy by model")
        st.bar_chart(model_accuracy.set_index("model"))

        trend_df = pd.read_sql(
            """
            SELECT
                DATE_FORMAT(resolved_at, '%%Y-%%m') AS month,
                AVG(outcome_correct) * 100 AS outcome_accuracy,
                AVG(score_exact) * 100 AS exact_score_accuracy
            FROM match_predictions
            WHERE status = 'resolved'
            GROUP BY DATE_FORMAT(resolved_at, '%%Y-%%m')
            ORDER BY month
            """,
            engine,
        )
        if not trend_df.empty:
            st.subheader("Accuracy trend by month")
            st.line_chart(trend_df.set_index("month"))

        recent_df = pd.read_sql(
            """
            SELECT
                m.match_date,
                t1.name AS home_team,
                t2.name AS away_team,
                mp.rf_prediction,
                mp.rf_confidence,
                mp.lr_prediction,
                mp.lr_confidence,
                mp.predicted_result,
                mp.actual_result,
                mp.outcome_correct,
                mp.score_exact
            FROM match_predictions mp
            JOIN matches m ON m.id = mp.match_id
            JOIN teams t1 ON m.home_team_id = t1.id
            JOIN teams t2 ON m.away_team_id = t2.id
            WHERE mp.status = 'resolved'
            ORDER BY mp.resolved_at DESC
            LIMIT 30
            """,
            engine,
        )
        if not recent_df.empty:
            outcome_map = {"H": "Home", "D": "Draw", "A": "Away"}
            recent_df["rf_outcome"] = recent_df["rf_prediction"].map(outcome_map)
            recent_df["lr_outcome"] = recent_df["lr_prediction"].map(outcome_map)
            recent_df["predicted_outcome"] = recent_df["predicted_result"].map(outcome_map)
            recent_df["actual_outcome"] = recent_df["actual_result"].map(outcome_map)
            recent_df["rf_confidence"] = (recent_df["rf_confidence"] * 100).round(1).astype(str) + "%"
            recent_df["lr_confidence"] = (recent_df["lr_confidence"] * 100).round(1).astype(str) + "%"
            recent_df["outcome_correct"] = recent_df["outcome_correct"].map({1: "✅", 0: "❌"})
            recent_df["score_exact"] = recent_df["score_exact"].map({1: "✅", 0: "❌"})
            st.subheader("Recent resolved predictions")
            st.dataframe(
                recent_df[
                    [
                        "match_date",
                        "home_team",
                        "away_team",
                        "rf_outcome",
                        "rf_confidence",
                        "lr_outcome",
                        "lr_confidence",
                        "predicted_outcome",
                        "actual_outcome",
                        "outcome_correct",
                        "score_exact",
                    ]
                ],
                use_container_width=True,
            )


elif page == "📈 Statistics":
    st.header("Team Statistics")

    query = """
        SELECT
            t.name as team,
            COUNT(*) as matches,
            SUM(CASE WHEN m.result = 'H' THEN 1 ELSE 0 END) as wins
        FROM matches m
        JOIN teams t ON m.home_team_id = t.id
        WHERE m.result IS NOT NULL
        GROUP BY t.id
        ORDER BY wins DESC
        LIMIT 10
    """
    df = pd.read_sql(query, engine)

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No statistics available")

elif page == "👤 Player Stats":
    st.header("Player Statistics & Availability")
    st.caption("Uses ESPN roster feed, including injuries/unavailability when provided.")

    league_slug = st.selectbox("League", ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"], index=0)
    teams_df = pd.read_sql("SELECT DISTINCT name FROM teams ORDER BY name ASC", engine)
    if teams_df.empty:
        st.info("No teams found in the local DB yet. Run data collection first.")
        selected_team = None
    else:
        default_team = "Barcelona" if "Barcelona" in teams_df["name"].values else teams_df["name"].iloc[0]
        selected_team = st.selectbox(
            "Team",
            teams_df["name"].tolist(),
            index=teams_df["name"].tolist().index(default_team),
        )

    if selected_team and st.button("Load player stats", type="primary"):
        try:
            report = get_team_player_report(league_slug, selected_team)
            players_df = report["players"]
            unavailable_df = report["unavailable"]

            if players_df.empty:
                st.warning("No player data returned for this team.")
            else:
                st.subheader("Squad stats")
                st.dataframe(
                    players_df.drop(columns=["injury_count"], errors="ignore")
                              .sort_values(["goals", "assists"], ascending=False),
                    use_container_width=True,
                )

            st.subheader("Unavailable / Injuries")
            if unavailable_df.empty:
                st.success("No unavailable players reported in current feed.")
            else:
                st.dataframe(unavailable_df, use_container_width=True)
        except requests.HTTPError as exc:
            st.error(f"Failed to fetch player feed: {exc}")
        except ValueError as exc:
            st.error(str(exc))

elif page == "🏥 Data Quality":
    st.header("Data Quality")

    query = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END) as finished
        FROM matches
    """
    df = pd.read_sql(query, engine)

    if not df.empty:
        row = df.iloc[0]
        st.metric("Total Matches", row["total"])
        st.metric("Finished Matches", row["finished"])

st.markdown("---")
st.markdown("Football Match Predictor | ML-powered predictions")
