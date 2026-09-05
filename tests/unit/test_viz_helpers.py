"""Pure-Python tests for the Streamlit dashboard's viz helper modules."""

import numpy as np
from scipy import stats

from grail.frame import Frame
from grail.viz.distributions_viz import default_support, pdf_values
from grail.viz.graph_viz import graph_to_plotly


def test_pdf_values_matches_scipy_beta():
    x = np.linspace(0.0, 1.0, 10)
    expected = stats.beta.pdf(x, 8.0, 2.0)

    result = pdf_values("beta", {"alpha": 8.0, "beta": 2.0}, x)

    assert np.allclose(result, expected)


def test_pdf_values_rejects_unknown_code():
    try:
        pdf_values("unknown", {}, np.array([0.0]))
    except ValueError as error:
        assert "unsupported distribution code" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown distribution code")


def test_default_support_beta_spans_unit_interval():
    x = default_support("beta", {"alpha": 8.0, "beta": 2.0})

    assert x.min() == 0.0
    assert x.max() == 1.0


def test_graph_to_plotly_has_expected_node_and_edge_counts(chain_frame: Frame):
    figure = graph_to_plotly(chain_frame)

    node_trace, edge_trace = figure.data[1], figure.data[0]
    assert len(node_trace.text) == 2  # Cause, Effect
    assert list(node_trace.text) == ["Cause", "Effect"]
    # One edge is drawn as a 3-point (x0, x1, None) segment.
    assert len(edge_trace.x) == 3
