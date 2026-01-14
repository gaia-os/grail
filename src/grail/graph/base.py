from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import uuid

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
    distribution_name: Optional[str] = None
    distribution_params: Dict[str, Any] = field(default_factory=dict)
    observations: Optional[Any] = None
    is_observed: bool = False
