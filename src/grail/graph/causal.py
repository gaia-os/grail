import networkx as nx
from typing import List, Optional, Union, Dict, Any
from grail.graph.base import Node, VariableNode, Edge

class CausalGraph:
    """
    A wrapper around networkx.DiGraph specialized for Causal Inference.
    Represents direct causal relationships between variables.
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
        
        self._graph.add_edge(u, v, **attrs)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieves the Node object by ID."""
        if self._graph.has_node(node_id):
            return self._graph.nodes[node_id].get('data')
        return None

    def get_parents(self, node_id: str) -> List[Node]:
        """Returns the parents (direct causes) of a node."""
        parent_ids = self._graph.predecessors(node_id)
        return [self.get_node(pid) for pid in parent_ids]

    def get_children(self, node_id: str) -> List[Node]:
        """Returns the children (direct effects) of a node."""
        child_ids = self._graph.successors(node_id)
        return [self.get_node(cid) for cid in child_ids]

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
