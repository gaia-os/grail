from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TypedDict, List
from grail.graph.causal import CausalGraph
from grail.graph.base import VariableNode
from grail.logger import logger


# total = False means all fields are optional
class FrameMetadata(TypedDict, total=False):
    """Metadata structure for Frame objects."""
    created_at: str
    updated_at: str
    description: str
    version: str
    tags: List[str]


@dataclass(eq=False, repr=False)
class Frame:
    """
    Frames house a mixture of data types and are essentially the containers/bounding-boxes on a world model
    """
    name: str
    graph: CausalGraph = field(default_factory=CausalGraph)
    metadata: FrameMetadata = field(default_factory=dict)

    def add_variable(self, name: str, dist: str, params: Dict[str, Any], observations: Optional[Any] = None) -> str:
        """Helper to add a variable to the frame's graph."""
        logger.debug(f"Adding variable '{name}' to Frame '{self.name}'. Distribution: {dist}")
        var = VariableNode(
            name=name,
            distribution_name=dist,
            distribution_params=params,
            observations=observations,
            is_observed=(observations is not None)
        )
        self.graph.add_node(var)
        return var.id

    def add_dependency(self, source_name_or_id: str, target_name_or_id: str):
        """Adds a causal dependency between two variables."""
        # Simple lookup strategy - if ID not found, assume name (this would need more robust lookup in production)
        
        # For now, let's assume usage of IDs or we search by name (inefficient O(N))
        # Let's rely on the caller passing IDs for now or implement name lookup on Graph
        
        # Note: The CausalGraph expects IDs or Node objects.
        # If strings are passed, we pass them through.
        self.graph.add_edge(source_name_or_id, target_name_or_id)
