import streamlit as st
from config import GAME, MEDIA_ACTION_MAP, ACTIVE, INACTIVE
import altair as alt
import pandas as pd

st.title(f"{GAME} Spotify Controller")
active_action_selection = st.selectbox("Action when game is acitve", options=MEDIA_ACTION_MAP[ACTIVE] )
lower_vol = st.slider(
    "Lower Volume to:",
    min_value=0.1,
    max_value=0.9,
    step=0.05,
    value=0.3,
    disabled=( active_action_selection != 'QUIET'),
)

frames_for_decision = st.slider(
    "Frames to evaluate for decision",
    min_value=2,
    max_value=16,
    step=2,
    value=4
)

#plot latency?
# st.altair_chart(
#     alt.Chart(pd.DataFrame()).mark_line().encode(
#         x=""
#     )
# )

# frame_view_cols = st.columns(frames_for_decision)
# for c in frame_view_cols:
#     c.empty()