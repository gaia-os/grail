"""Browse registered Frames: metadata, variables, causal graph, and posteriors."""

from lib import get_posterior, list_frame_records, load_frame
import numpy as np
import streamlit as st

from grail.viz.distributions import default_support, pdf_values
from grail.viz.graph import graph_to_plotly

st.title("Frames")

records = list_frame_records()
if not records:
    st.info("No Frames are registered yet.")
    st.stop()

names = [record.name for record in records]
selected_name = st.selectbox("Frame", names)
record = next(record for record in records if record.name == selected_name)
try:
    frame = load_frame(record.spec_path)
except FileNotFoundError:
    st.error(f"Registered spec file for '{selected_name}' is missing: {record.spec_path}")
    st.stop()

st.subheader("Metadata")
st.write(f"**Description:** {frame.metadata.description or '_none_'}")
st.write(f"**Tags:** {', '.join(frame.metadata.tags) if frame.metadata.tags else '_none_'}")

st.subheader("Variables")
variables = frame.get_variables()
st.dataframe(
    [
        {
            "name": variable.name,
            "distribution": variable.get_distribution_name(),
            "params": variable.get_distribution_params(),
            "observed": variable.is_observed(),
        }
        for variable in variables
    ],
    width="stretch",
)

st.subheader("Dependency graph")
st.plotly_chart(graph_to_plotly(frame), width="stretch")

st.subheader("Posteriors")
any_posterior = False
for variable in variables:
    posterior = get_posterior(frame, variable.name)
    if posterior is None:
        continue
    any_posterior = True
    st.write(f"**{variable.name}** — {posterior.distribution}({posterior.params})")
    x = default_support(posterior.distribution, posterior.params)
    y = pdf_values(posterior.distribution, posterior.params, np.asarray(x))
    st.line_chart({"x": x, "density": y}, x="x", y="density")

if not any_posterior:
    st.info("No posteriors have been persisted for this Frame yet.")
