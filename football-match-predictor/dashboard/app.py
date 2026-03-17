from pathlib import Path
import sys

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from db import engine

st.title("Football Match Predictor")
st.write("Simple starter dashboard")

try:
    teams_df = pd.read_sql("SELECT * FROM teams", engine)
    matches_df = pd.read_sql("SELECT * FROM matches", engine)

    st.subheader("Teams")
    st.dataframe(teams_df, use_container_width=True)

    st.subheader("Matches")
    st.dataframe(matches_df, use_container_width=True)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.info("Run src/init_db.py first and check your .env settings.")
