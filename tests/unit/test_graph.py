"""DAG construction and traversal in the causal graph wrapper."""


import pytest

from grail.frame.variable import Variable
from grail.graph.base import Node, VariableNode
from grail.graph.causal import CausalGraph


def _variable_node(name: str) -> VariableNode:
    return VariableNode(name=name, variable=Variable(name=name))


def test_nodes_are_stored_and_retrieved_by_id():
    graph = CausalGraph()
    node = _variable_node("Cause")

    graph.add_node(node)

    assert graph.get_node(node.id) is node
    assert graph.get_node("absent") is None


def test_edges_accept_node_objects_and_bare_ids():
    graph = CausalGraph()
    cause, effect, other = (_variable_node(n) for n in ("Cause", "Effect", "Other"))
    for node in (cause, effect, other):
        graph.add_node(node)

    graph.add_edge(cause, effect)
    graph.add_edge(effect.id, other.id)

    assert graph.graph.has_edge(cause.id, effect.id)
    assert graph.graph.has_edge(effect.id, other.id)


def test_edges_must_connect_existing_nodes():
    graph = CausalGraph()
    node = _variable_node("Lonely")
    graph.add_node(node)

    with pytest.raises(KeyError, match="must connect existing graph nodes"):
        graph.add_edge(node.id, "missing")


def test_self_edges_are_rejected():
    graph = CausalGraph()
    node = _variable_node("Loop")
    graph.add_node(node)

    with pytest.raises(ValueError, match="cannot target the same node"):
        graph.add_edge(node.id, node.id)


def test_a_rejected_cycle_leaves_the_graph_unchanged():
    graph = CausalGraph()
    cause, effect = _variable_node("Cause"), _variable_node("Effect")
    graph.add_node(cause)
    graph.add_node(effect)
    graph.add_edge(cause.id, effect.id)

    with pytest.raises(ValueError, match="directed acyclic graph"):
        graph.add_edge(effect.id, cause.id)

    assert graph.graph.number_of_edges() == 1
    assert not graph.graph.has_edge(effect.id, cause.id)


def test_parents_and_children_resolve_to_node_objects():
    graph = CausalGraph()
    cause, effect = _variable_node("Cause"), _variable_node("Effect")
    graph.add_node(cause)
    graph.add_node(effect)
    graph.add_edge(cause.id, effect.id)

    assert graph.get_children(cause.id) == [effect]
    assert graph.get_parents(effect.id) == [cause]
    assert graph.get_parents(cause.id) == []


def test_get_variables_ignores_non_variable_nodes():
    graph = CausalGraph()
    variable = _variable_node("Cause")
    graph.add_node(variable)
    graph.add_node(Node(name="annotation"))

    assert graph.get_variables() == [variable]


def test_topological_sort_orders_parents_before_children():
    graph = CausalGraph()
    first, second, third = (_variable_node(n) for n in ("First", "Second", "Third"))
    for node in (third, first, second):
        graph.add_node(node)
    graph.add_edge(first.id, second.id)
    graph.add_edge(second.id, third.id)

    order = graph.topological_sort()

    assert order.index(first.id) < order.index(second.id) < order.index(third.id)


def test_edge_attributes_are_retained():
    graph = CausalGraph()
    cause, effect = _variable_node("Cause"), _variable_node("Effect")
    graph.add_node(cause)
    graph.add_node(effect)

    graph.add_edge(cause.id, effect.id, mechanism="linear")

    assert graph.graph.edges[cause.id, effect.id]["mechanism"] == "linear"
