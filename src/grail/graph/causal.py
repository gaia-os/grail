from typing import List, Optional, Union

import networkx as nx

from grail.graph.base import Node, VariableNode


class CausalGraph:
    """
    A wrapper around networkx.DiGraph specialized for Causal Inference.
    Represents direct causal relationships between variables.

    Note: this class stores each runtime node object under the node attribute
    key ``"data"``. That key is a local GRAIL convention (not special to
    NetworkX), and values only need to be serializable if you export/persist
    the raw graph structure.
    """

    def __init__(self):
        self._graph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying networkx graph."""
        return self._graph

    def add_node(self, node: Node):
        """Adds a node to the causal graph."""
        self._graph.add_node(node.id, data=node)

    def add_edge(self, source: Union[Node, str], target: Union[Node, str], **attrs):
        """
        Adds a directed edge representing a causal link.
        source/target can be Node objects or their IDs.
        """
        u = source.id if isinstance(source, Node) else source
        v = target.id if isinstance(target, Node) else target
        if not self._graph.has_node(u) or not self._graph.has_node(v):
            raise KeyError("causal dependencies must connect existing graph nodes")
        if u == v:
            raise ValueError("a causal dependency cannot target the same node")
        self._graph.add_edge(u, v, **attrs)
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(u, v)
            raise ValueError("causal dependencies must form a directed acyclic graph")

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieves the Node object by ID."""
        if self._graph.has_node(node_id):
            return self._graph.nodes[node_id].get('data')
        return None

    def get_parents(self, node_id: str) -> List[Node]:
        """Returns the parents (direct causes) of a node."""
        parent_ids = self._graph.predecessors(node_id)
        return [node for pid in parent_ids if (node := self.get_node(pid)) is not None]

    def get_children(self, node_id: str) -> List[Node]:
        """Returns the children (direct effects) of a node."""
        child_ids = self._graph.successors(node_id)
        return [node for cid in child_ids if (node := self.get_node(cid)) is not None]

    def get_variables(self) -> List[VariableNode]:
        """Returns all nodes that are of type VariableNode."""
        vars = []
        for _, data in self._graph.nodes(data=True):
            node = data.get('data')
            if isinstance(node, VariableNode):
                vars.append(node)
        return vars

    def topological_sort(self) -> List[str]:
        """Returns a list of node IDs in topological order."""
        return list(nx.topological_sort(self._graph))
