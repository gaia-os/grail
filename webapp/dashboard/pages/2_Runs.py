"""Browse persisted runs: metadata, diagnostics, and posterior results."""

import json
from pathlib import Path

from lib import list_runs
import numpy as np
import streamlit as st

from grail.viz.distributions_viz import default_support, pdf_values

st.title("Runs")

runs = list_runs()
if not runs:
    st.info("No runs have been persisted yet.")
    st.stop()

st.dataframe(
    [
        {
            "id": run.id,
            "frame_name": run.frame_name,
            "strategy_id": run.strategy_id,
            "operation_kind": run.operation_kind,
            "status": run.status.value,
            "created_at": run.created_at,
        }
        for run in runs
    ],
    width="stretch",
)

run_ids = [run.id for run in runs]
selected_id = st.selectbox("Run", run_ids)
run = next(run for run in runs if run.id == selected_id)

st.subheader("Run record")
st.json(run.to_dict())

results_path = run.artifact_paths.get("results")
if results_path and Path(results_path).exists():
    st.subheader("Posteriors")
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    for variable_name, posterior in results.get("posteriors", {}).items():
        st.write(f"**{variable_name}** — {posterior['distribution']}({posterior['params']})")
        x = default_support(posterior["distribution"], posterior["params"])
        y = pdf_values(posterior["distribution"], posterior["params"], np.asarray(x))
        st.line_chart({"x": x, "density": y}, x="x", y="density")
