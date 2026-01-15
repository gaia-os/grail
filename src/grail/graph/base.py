from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import uuid

from grail.stats.variable import Variable

@dataclass
class Node:
    """Base class for any node in the GRAIL system."""
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class Edge:
    """Base class for an edge between nodes."""
    source: str  # Node ID
    target: str  # Node ID
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VariableNode(Node):
    """
    A Variable type Node.
    Has an associated GRAIL Variable
    """
    variable: Variable

    @property
    def is_observed(self) -> bool:
        return self.variable.is_observed()

    def get_distribution_name(self) -> Optional[str]:
        return self.variable.get_distribution_name()

    def get_distribution_params(self) -> Dict[str, Any]:
        return self.variable.get_distribution_params()

    def get_observations(self) -> Optional[Any]:
        """
        Return observations stored on the underlying Variable, if any.
        """
        return self.variable.get_observations()
