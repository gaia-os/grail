from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from grail.graph.causal import CausalGraph
from grail.graph.base import Variable
from grail.logger import logger

@dataclass
class Frame:
    """
    A Frame is a staging area for a simulation.
    It contains the causal graph (world model), data, and configuration.
    """
    name: str
    graph: CausalGraph = field(default_factory=CausalGraph)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Simulation parameters could be stored here
    sim_config: Dict[str, Any] = field(default_factory=dict)

    def add_variable(self, name: str, dist: str, params: Dict[str, Any], observed: Optional[Any] = None) -> str:
        """Helper to add a variable to the frame's graph."""
        logger.debug(f"Adding variable '{name}' to Frame '{self.name}'. Distribution: {dist}")
        var = Variable(
            name=name,
            distribution_name=dist,
            distribution_params=params,
            observed_value=observed,
            is_observed=(observed is not None)
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
