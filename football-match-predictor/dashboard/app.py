"""
Football Match Predictor Dashboard

A Streamlit dashboard for viewing match predictions, statistics, and insights.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import joblib

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from db import engine
from generate_features import generate_features_for_match


# Page config
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ Football Match Predictor")
st.markdown("*Powered by Machine Learning & Historical Data*")
st.divider()


def load_model():
    """Load the trained ML model."""
    model_path = project_root / "models" / "model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def get_upcoming_matches():
    """Get all upcoming matches with team names."""
    query = """
        SELECT 
            m.id,
            m.match_date,
            m.home_team_id,
            m.away_team_id,
            t1.name as home_team,
            t2.name as away_team
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.result IS NULL
        ORDER BY m.match_date ASC
    """
    return pd.read_sql(query, engine)


def get_recent_matches(limit=10):
    """Get recent finished matches."""
    query = f"""
        SELECT 
            m.match_date,
            t1.name as home_team,
            m.home_goals,
            m.away_goals,
            t2.name as away_team,
            m.result
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.result IS NOT NULL
        ORDER BY m.match_date DESC
        LIMIT {limit}
    """
    return pd.read_sql(query, engine)


def get_statistics():
    """Get overall statistics."""
    query = """
        SELECT 
            COUNT(*) as total_matches,
            COUNT(CASE WHEN result = 'H' THEN 1 END) as home_wins,
            COUNT(CASE WHEN result = 'D' THEN 1 END) as draws,
            COUNT(CASE WHEN result = 'A' THEN 1 END) as away_wins,
            COUNT(CASE WHEN result IS NULL THEN 1 END) as upcoming
        FROM matches
    """
    return pd.read_sql(query, engine).iloc[0]


def predict_match(model, match):
    """Make prediction for a single match."""
    features = generate_features_for_match(
        match['id'],
        match['match_date'],
        match['home_team_id'],
        match['away_team_id']
    )
    
    X = pd.DataFrame([{
        'home_last5_points': features['home_last5_points'],
        'home_last5_goal_diff': features['home_last5_goal_diff'],
        'home_form_pct': features['home_form_pct'],
        'away_last5_points': features['away_last5_points'],
        'away_last5_goal_diff': features['away_last5_goal_diff'],
        'away_form_pct': features['away_form_pct'],
        'h2h_home_wins': features['h2h_home_wins'],
        'h2h_draws': features['h2h_draws'],
        'h2h_away_wins': features['h2h_away_wins']
    }])
    
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    
    return prediction, prob_dict, features


try:
    # Load model
    model = load_model()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "📈 Recent Matches", "ℹ️ About"])
    
    # TAB 1: Overview
    with tab1:
        st.header("Database Statistics")
        
        stats = get_statistics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Matches", stats['total_matches'])
        col2.metric("Home Wins", stats['home_wins'], 
                    f"{stats['home_wins']/stats['total_matches']*100:.1f}%")
        col3.metric("Draws", stats['draws'], 
                    f"{stats['draws']/stats['total_matches']*100:.1f}%")
        col4.metric("Away Wins", stats['away_wins'], 
                    f"{stats['away_wins']/stats['total_matches']*100:.1f}%")
        col5.metric("Upcoming", stats['upcoming'])
        
        st.divider()
        
        # Show teams
        st.subheader("Teams in Database")
        teams_df = pd.read_sql("SELECT id, name FROM teams ORDER BY name", engine)
        st.dataframe(teams_df, use_container_width=True, hide_index=True)
    
    # TAB 2: Predictions
    with tab2:
        st.header("Match Predictions")
        
        if model is None:
            st.warning("⚠️ Model not trained yet!")
            st.info("Run: `python src/train_model.py` to train the model first.")
        else:
            upcoming = get_upcoming_matches()
            
            if upcoming.empty:
                st.info("No upcoming matches found. Run `python src/collect_data.py` to fetch matches.")
            else:
                st.success(f"Found {len(upcoming)} upcoming match(es)")
                
                for idx, match in upcoming.iterrows():
                    with st.container():
                        st.subheader(f"📅 {match['match_date'].strftime('%A, %B %d, %Y at %H:%M')}")
                        
                        # Make prediction
                        prediction, prob_dict, features = predict_match(model, match)
                        
                        # Display match
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            st.markdown(f"### {match['home_team']}")
                            st.metric("Recent Form", f"{features['home_form_pct']:.1f}%")
                            st.caption(f"Last 5: {features['home_last5_points']} pts, {features['home_last5_goal_diff']:+d} GD")
                        
                        with col2:
                            st.markdown("### VS")
                            outcome_map = {'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'}
                            st.info(f"**Prediction:**\n\n{outcome_map[prediction]}")
                        
                        with col3:
                            st.markdown(f"### {match['away_team']}")
                            st.metric("Recent Form", f"{features['away_form_pct']:.1f}%")
                            st.caption(f"Last 5: {features['away_last5_points']} pts, {features['away_last5_goal_diff']:+d} GD")
                        
                        # Probabilities
                        st.markdown("#### Prediction Probabilities")
                        prob_cols = st.columns(3)
                        prob_cols[0].metric("🏠 Home Win", f"{prob_dict.get('H', 0):.1%}")
                        prob_cols[1].metric("🤝 Draw", f"{prob_dict.get('D', 0):.1%}")
                        prob_cols[2].metric("✈️ Away Win", f"{prob_dict.get('A', 0):.1%}")
                        
                        # Head-to-head
                        h2h_total = features['h2h_home_wins'] + features['h2h_draws'] + features['h2h_away_wins']
                        if h2h_total > 0:
                            st.markdown("#### Head-to-Head History")
                            h2h_cols = st.columns(3)
                            h2h_cols[0].metric("Home Wins", features['h2h_home_wins'])
                            h2h_cols[1].metric("Draws", features['h2h_draws'])
                            h2h_cols[2].metric("Away Wins", features['h2h_away_wins'])
                        
                        st.divider()
    
    # TAB 3: Recent Matches
    with tab3:
        st.header("Recent Match Results")
        
        recent = get_recent_matches(20)
        
        if recent.empty:
            st.info("No match results yet.")
        else:
            # Format the dataframe
            recent['match_date'] = pd.to_datetime(recent['match_date']).dt.strftime('%Y-%m-%d %H:%M')
            recent['score'] = recent['home_goals'].astype(str) + ' - ' + recent['away_goals'].astype(str)
            recent['outcome'] = recent['result'].map({'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'})
            
            display_df = recent[['match_date', 'home_team', 'score', 'away_team', 'outcome']]
            display_df.columns = ['Date', 'Home Team', 'Score', 'Away Team', 'Result']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # TAB 4: About
    with tab4:
        st.header("About This Dashboard")
        
        st.markdown("""
        ### 🎯 What This Does
        
        This dashboard uses **machine learning** to predict football match outcomes based on:
        - Recent team form (last 5 matches)
        - Goal difference trends
        - Head-to-head history
        - Home/away performance
        
        ### 🧠 How It Works
        
        1. **Data Collection** (`collect_data.py`)
           - Fetches match data from ESPN API
           - Validates and stores in MySQL database
           - Prevents duplicates automatically
        
        2. **Feature Engineering** (`generate_features.py`)
           - Calculates team statistics from historical matches
           - Creates predictive features WITHOUT data leakage
           - Only uses information available BEFORE each match
        
        3. **Model Training** (`train_model.py`)
           - Trains Random Forest classifier on 778 matches
           - Achieves ~45% accuracy (realistic for football!)
           - Uses train/test split for proper evaluation
        
        4. **Predictions** (`predict_upcoming.py`)
           - Generates probabilities for H/D/A outcomes
           - Shows team form and head-to-head stats
        
        ### 📊 Model Performance
        
        - **Accuracy:** ~45% (football is unpredictable!)
        - **Best at:** Home win predictions
        - **Struggles with:** Draw predictions (very hard to predict)
        - **Key features:** Goal difference and recent form
        
        ### 🚀 How to Update
        
        ```bash
        # Fetch latest matches
        python src/collect_data.py
        
        # Regenerate features
        python src/generate_features.py
        
        # Retrain model
        python src/train_model.py
        
        # Refresh this dashboard
        ```
        
        ### ⚠️ Disclaimer
        
        These predictions are for **educational purposes** only. Football is inherently 
        unpredictable, and no model can guarantee accurate predictions. Use as guidance, 
        not as betting advice!
        """)
        
        st.divider()
        
        st.markdown("### 📚 Data Sources")
        st.markdown("- Match data: ESPN API")
        st.markdown("- Database: MySQL (via XAMPP)")
        st.markdown("- ML Framework: scikit-learn")

except Exception as exc:
    st.error(f"❌ Error: {exc}")
    st.info("""
    **Troubleshooting:**
    1. Make sure XAMPP MySQL is running
    2. Run `python src/init_db.py` to initialize database
    3. Run `python src/collect_data.py` to fetch matches
    4. Run `python src/generate_features.py` to create features
    5. Run `python src/train_model.py` to train the model
    """)
