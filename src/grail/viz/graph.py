"""Render a Frame's causal graph as a Plotly figure."""

from typing import TYPE_CHECKING

import networkx as nx
import plotly.graph_objects as go

if TYPE_CHECKING:
    from grail.frame import Frame


def graph_to_plotly(frame: "Frame") -> go.Figure:
    """Return a Plotly figure of a Frame's variable dependency graph."""
    graph = frame.graph.graph
    layout = nx.spring_layout(graph, seed=0)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []
    node_labels = []
    for node_id in graph.nodes():
        variable = frame.graph.get_node(node_id).variable
        x, y = layout[node_id]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(variable.name)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line={"width": 1, "color": "#888"}, hoverinfo="none"
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        marker={"size": 18, "color": "#1f77b4"},
        hoverinfo="text",
    )
    figure = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Frame '{frame.name}' dependency graph",
            showlegend=False,
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        ),
    )
    return figure
