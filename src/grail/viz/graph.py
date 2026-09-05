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

    edge_annotations: list[go.layout.Annotation] = []
    for source, target in graph.edges():
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_annotations.append(
            go.layout.Annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#888",
                arrowside="end+start",
                startarrowhead=0,
                standoff=10,
                startstandoff=10,
                showarrow=True,
            )
        )

    node_x = []
    node_y = []
    node_labels = []
    for node_id in graph.nodes():
        variable = frame.graph.get_node(node_id).variable
        x, y = layout[node_id]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(variable.name)

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
        data=[node_trace],
        layout=go.Layout(
            title=f"Frame '{frame.name}' dependency graph",
            showlegend=False,
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            annotations=edge_annotations,
        ),
    )
    return figure
