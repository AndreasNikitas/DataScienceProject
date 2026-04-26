"""
Football Match Predictor — Dashboard
"""

import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db import engine
from player_stats import get_team_player_report

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* prediction card outcome box */
    .outcome-box {
        background: #1e3a5f;
        color: white;
        border-radius: 10px;
        padding: 20px 12px;
        text-align: center;
    }
    .outcome-label {
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .outcome-sub {
        font-size: 12px;
        opacity: 0.75;
        margin: 2px 0;
    }

    /* team name in prediction row */
    .team-name {
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        margin: 6px 0 2px 0;
    }
    .team-role {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin: 0;
    }
    .match-date {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-bottom: 10px;
    }

    /* page header */
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 2px;
    }
    .page-sub {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ── Navigation ─────────────────────────────────────────────────
PAGES = ["Overview", "Predictions", "Results", "Teams", "Players"]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

OUTCOME_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


# ═══════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<p class="page-title">⚽ Football Match Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">ML-powered predictions using Random Forest & Logistic Regression</p>', unsafe_allow_html=True)

    # ── Database stats ──
    total   = pd.read_sql("SELECT COUNT(*) AS n FROM matches", engine).iloc[0]["n"]
    teams   = pd.read_sql("SELECT COUNT(*) AS n FROM teams", engine).iloc[0]["n"]
    finished = pd.read_sql("SELECT COUNT(*) AS n FROM matches WHERE result IS NOT NULL", engine).iloc[0]["n"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Matches", int(total))
    c2.metric("Teams", int(teams))
    c3.metric("Finished Matches", int(finished))

    st.markdown("---")

    # ── Prediction accuracy ──
    acc = pd.read_sql("""
        SELECT
            COUNT(*) AS resolved,
            AVG(recent.outcome_correct) * 100 AS outcome_pct,
            AVG(CASE WHEN recent.rf_prediction = recent.actual_result THEN 1 ELSE 0 END) * 100 AS rf_pct,
            AVG(CASE WHEN recent.lr_prediction = recent.actual_result THEN 1 ELSE 0 END) * 100 AS lr_pct
        FROM (
            SELECT mp.*
            FROM (
                SELECT mp1.*
                FROM match_predictions mp1
                JOIN (
                    SELECT match_id, MAX(id) AS id
                    FROM match_predictions
                    WHERE status = 'resolved'
                    GROUP BY match_id
                ) latest
                  ON latest.id = mp1.id
            ) mp
            ORDER BY mp.resolved_at DESC, mp.id DESC
            LIMIT 30
        ) recent
    """, engine).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions Resolved", int(acc["resolved"] or 0))
    c2.metric("Overall Accuracy",     f"{float(acc['outcome_pct'] or 0):.1f}%")
    c3.metric("Random Forest",        f"{float(acc['rf_pct'] or 0):.1f}%")
    c4.metric("Logistic Regression",  f"{float(acc['lr_pct'] or 0):.1f}%")

    st.markdown("---")

    # ── How it works ──
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Data Collection**")
        st.write("Match results and upcoming fixtures are pulled from the ESPN API and stored in a MySQL database.")
    with c2:
        st.markdown("**2. Feature Engineering**")
        st.write("28 numbers are calculated per match — team form, ELO rating, rest days, head-to-head record, and more.")
    with c3:
        st.markdown("**3. Prediction**")
        st.write("Two ML models (Random Forest + Logistic Regression) independently predict the outcome. Their agreement or disagreement is shown.")


# ═══════════════════════════════════════════════════════════════
# PREDICTIONS
# ═══════════════════════════════════════════════════════════════
elif page == "Predictions":
    st.markdown('<p class="page-title">Upcoming Predictions</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">What the models predict for matches yet to be played</p>', unsafe_allow_html=True)

    # Warn if matches are stuck without a result
    stale = int(pd.read_sql("""
        SELECT COUNT(*) AS n FROM matches
        WHERE result IS NULL AND match_date < UTC_TIMESTAMP()
    """, engine).iloc[0]["n"])
    if stale > 0:
        st.warning(f"{stale} match(es) are in the past with no result yet — run: python3 src/collect_data.py")

    unpredicted = int(pd.read_sql("""
        SELECT COUNT(*) AS n
        FROM matches m
        LEFT JOIN match_predictions mp ON mp.match_id = m.id AND mp.status = 'pending'
        WHERE m.result IS NULL AND m.match_date >= UTC_TIMESTAMP() AND mp.id IS NULL
    """, engine).iloc[0]["n"])
    if unpredicted > 0:
        st.warning(f"{unpredicted} upcoming match(es) have no predictions yet — run: python3 src/predict_upcoming.py")

    df = pd.read_sql("""
        SELECT
            m.match_date,
            t1.name AS home_team,
            t2.name AS away_team,
            mp.predicted_result,
            mp.predicted_home_goals,
            mp.predicted_away_goals,
            mp.rf_prediction,
            mp.rf_confidence,
            mp.lr_prediction,
            mp.lr_confidence
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        JOIN (
            SELECT mp1.*
            FROM match_predictions mp1
            JOIN (
                SELECT match_id, MAX(id) AS id
                FROM match_predictions
                WHERE status = 'pending'
                GROUP BY match_id
            ) latest ON latest.id = mp1.id
        ) mp ON mp.match_id = m.id
        WHERE m.result IS NULL AND m.match_date >= UTC_TIMESTAMP()
        ORDER BY m.match_date ASC
        LIMIT 50
    """, engine)

    if df.empty:
        st.info("No upcoming predictions found. Run: python3 src/predict_upcoming.py")
    else:
        for _, row in df.iterrows():
            date_str  = pd.to_datetime(row["match_date"]).strftime("%a %d %b %Y · %H:%M")
            predicted = OUTCOME_MAP.get(row["predicted_result"], "—")
            rf_conf   = f"{float(row['rf_confidence'] or 0) * 100:.0f}%" if row["rf_confidence"] else "—"
            lr_conf   = f"{float(row['lr_confidence'] or 0) * 100:.0f}%" if row["lr_confidence"] else "—"
            agreed    = row["rf_prediction"] == row["lr_prediction"]
            agreement = "✓ Both models agree" if agreed else "⚠ Models disagree"

            col_home, col_mid, col_away = st.columns([2, 2, 2])

            with col_home:
                st.markdown(f"<p class='match-date'>{date_str}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='team-name'>{row['home_team']}</p>", unsafe_allow_html=True)
                st.markdown("<p class='team-role'>HOME</p>", unsafe_allow_html=True)

            with col_mid:
                st.markdown(f"""
                <div class="outcome-box">
                    <p class="outcome-label">{predicted}</p>
                    <p class="outcome-sub">{agreement}</p>
                    <p class="outcome-sub">RF {rf_conf} &nbsp;·&nbsp; LR {lr_conf}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_away:
                st.markdown("<p class='match-date'>&nbsp;</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='team-name'>{row['away_team']}</p>", unsafe_allow_html=True)
                st.markdown("<p class='team-role'>AWAY</p>", unsafe_allow_html=True)

            st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:12px 0'>",
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "Results":
    st.markdown('<p class="page-title">Prediction Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">How accurate were the models on matches that have already been played?</p>', unsafe_allow_html=True)

    acc = pd.read_sql("""
        SELECT
            COUNT(*) AS resolved,
            AVG(outcome_correct) * 100 AS outcome_pct,
            AVG(CASE WHEN rf_prediction = actual_result THEN 1 ELSE 0 END) * 100 AS rf_pct,
            AVG(CASE WHEN lr_prediction = actual_result THEN 1 ELSE 0 END) * 100 AS lr_pct
        FROM match_predictions mp
        JOIN (
            SELECT match_id, MAX(id) AS id
            FROM match_predictions
            WHERE status = 'resolved'
            GROUP BY match_id
        ) latest ON latest.id = mp.id
    """, engine).iloc[0]

    resolved = int(acc["resolved"] or 0)

    if resolved == 0:
        st.info("No resolved predictions yet. Run: python3 src/backtest.py")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches Resolved",   resolved)
        c2.metric("Overall Accuracy",   f"{float(acc['outcome_pct'] or 0):.1f}%")
        c3.metric("Random Forest",      f"{float(acc['rf_pct'] or 0):.1f}%")
        c4.metric("Logistic Regression", f"{float(acc['lr_pct'] or 0):.1f}%")

        st.markdown("---")

        # Model accuracy bar chart
        st.subheader("Model Comparison")
        chart_df = pd.DataFrame({
            "Model": ["Random Forest", "Logistic Regression"],
            "Accuracy (%)": [float(acc["rf_pct"] or 0), float(acc["lr_pct"] or 0)],
        }).set_index("Model")
        st.bar_chart(chart_df)

        st.markdown("---")

        # Recent predictions table
        recent = pd.read_sql("""
            SELECT
                m.match_date,
                t1.name AS home_team,
                t2.name AS away_team,
                mp.predicted_result,
                mp.actual_result,
                mp.rf_confidence,
                mp.lr_confidence,
                mp.outcome_correct
            FROM matches m
            JOIN teams t1 ON t1.id = m.home_team_id
            JOIN teams t2 ON t2.id = m.away_team_id
            JOIN (
                SELECT mp1.*
                FROM match_predictions mp1
                JOIN (
                    SELECT match_id, MAX(id) AS id
                    FROM match_predictions
                    WHERE status = 'resolved'
                    GROUP BY match_id
                ) latest
                  ON latest.id = mp1.id
            ) mp ON mp.match_id = m.id
            ORDER BY mp.resolved_at DESC, mp.id DESC
        """, engine)

        if not recent.empty:
            st.subheader("Recent Predictions")
            recent["Date"]      = pd.to_datetime(recent["match_date"]).dt.strftime("%d %b %Y")
            recent["Predicted"] = recent["predicted_result"].map(OUTCOME_MAP)
            recent["Actual"]    = recent["actual_result"].map(OUTCOME_MAP)
            recent["RF Conf"]   = (recent["rf_confidence"] * 100).round(1).astype(str) + "%"
            recent["LR Conf"]   = (recent["lr_confidence"] * 100).round(1).astype(str) + "%"
            recent["Correct"]   = recent["outcome_correct"].map({1: "✅", 0: "❌"})

            st.dataframe(
                recent[["Date", "home_team", "away_team", "Predicted", "Actual",
                         "RF Conf", "LR Conf", "Correct"]]
                .rename(columns={"home_team": "Home Team", "away_team": "Away Team"}),
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════
# TEAMS
# ═══════════════════════════════════════════════════════════════
elif page == "Teams":
    st.markdown('<p class="page-title">Team Statistics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Season record and recent results for any team in the database</p>', unsafe_allow_html=True)

    teams_df = pd.read_sql("SELECT name FROM teams ORDER BY name ASC", engine)
    if teams_df.empty:
        st.info("No teams found. Run: python3 src/collect_data.py")
    else:
        selected = st.selectbox("Select a team", teams_df["name"].tolist())

        # Season stats
        stats = pd.read_sql(f"""
            SELECT
                COUNT(*) AS matches,
                SUM(CASE
                    WHEN (home_team_id = t.id AND result = 'H')
                      OR (away_team_id = t.id AND result = 'A') THEN 1 ELSE 0
                END) AS wins,
                SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) AS draws,
                SUM(CASE
                    WHEN (home_team_id = t.id AND result = 'A')
                      OR (away_team_id = t.id AND result = 'H') THEN 1 ELSE 0
                END) AS losses,
                SUM(CASE WHEN home_team_id = t.id THEN home_goals ELSE away_goals END) AS scored,
                SUM(CASE WHEN home_team_id = t.id THEN away_goals ELSE home_goals END) AS conceded
            FROM matches m
            JOIN teams t ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            WHERE t.name = '{selected}' AND m.result IS NOT NULL
        """, engine).iloc[0]

        matches = int(stats["matches"] or 0)
        wins    = int(stats["wins"]    or 0)
        win_rate = (wins / matches * 100) if matches > 0 else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Matches",  matches)
        c2.metric("Wins",     wins)
        c3.metric("Draws",    int(stats["draws"]    or 0))
        c4.metric("Losses",   int(stats["losses"]   or 0))
        c5.metric("Win Rate", f"{win_rate:.1f}%")
        c6.metric("Scored / Conceded", f"{int(stats['scored'] or 0)} / {int(stats['conceded'] or 0)}")

        st.markdown("---")

        # Last 10 results
        form = pd.read_sql(f"""
            SELECT
                m.match_date,
                t1.name AS home_team,
                t2.name AS away_team,
                m.home_goals,
                m.away_goals,
                m.result,
                CASE WHEN m.home_team_id = t.id THEN 'Home' ELSE 'Away' END AS venue
            FROM matches m
            JOIN teams t  ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            JOIN teams t1 ON m.home_team_id = t1.id
            JOIN teams t2 ON m.away_team_id = t2.id
            WHERE t.name = '{selected}' AND m.result IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT 10
        """, engine)

        if not form.empty:
            st.subheader("Last 10 Results")

            def get_outcome(row):
                if row["venue"] == "Home":
                    return "W" if row["result"] == "H" else ("D" if row["result"] == "D" else "L")
                return "W" if row["result"] == "A" else ("D" if row["result"] == "D" else "L")

            form["W/D/L"] = form.apply(get_outcome, axis=1)
            form["Score"] = form["home_goals"].astype(str) + " - " + form["away_goals"].astype(str)
            form["Date"]  = pd.to_datetime(form["match_date"]).dt.strftime("%d %b %Y")

            st.dataframe(
                form[["Date", "home_team", "away_team", "Score", "venue", "W/D/L"]]
                .rename(columns={"home_team": "Home", "away_team": "Away", "venue": "Venue"}),
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════
# PLAYERS
# ═══════════════════════════════════════════════════════════════
elif page == "Players":
    st.markdown('<p class="page-title">Player Availability</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Squad stats and unavailable players from the ESPN roster feed</p>', unsafe_allow_html=True)

    league_slug = st.selectbox("League", ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"])
    teams_df = pd.read_sql("SELECT name FROM teams ORDER BY name ASC", engine)

    if teams_df.empty:
        st.info("No teams found. Run: python3 src/collect_data.py")
    else:
        selected = st.selectbox("Team", teams_df["name"].tolist())

        if st.button("Load", type="primary"):
            try:
                report         = get_team_player_report(league_slug, selected)
                players_df     = report["players"]
                unavailable_df = report["unavailable"]

                if not players_df.empty:
                    st.subheader("Squad")
                    st.dataframe(
                        players_df.drop(columns=["injury_count"], errors="ignore")
                                  .sort_values(["goals", "assists"], ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.subheader("Unavailable Players")
                if unavailable_df.empty:
                    st.success("No unavailable players reported.")
                else:
                    st.dataframe(unavailable_df, use_container_width=True, hide_index=True)

            except requests.HTTPError as e:
                st.error(f"Could not fetch player data: {e}")
            except ValueError as e:
                st.error(str(e))

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:13px'>"
    "Football Match Predictor · Random Forest & Logistic Regression · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
