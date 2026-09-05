"""GRAIL dashboard entry point."""

from lib import list_frame_records, list_runs
import streamlit as st

st.set_page_config(page_title="GRAIL Dashboard", layout="wide")

st.title("GRAIL Dashboard")
st.write(
    "Browse registered Frames and their causal graphs, and inspect persisted "
    "inference runs and posteriors. Use the sidebar to navigate."
)

frames = list_frame_records()
runs = list_runs()

col1, col2 = st.columns(2)
col1.metric("Registered Frames", len(frames))
col2.metric("Persisted Runs", len(runs))
