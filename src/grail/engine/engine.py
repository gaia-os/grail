from typing import Callable, Dict, Any, List
import pyro
import torch
from grail.builder.frame import Frame
from grail.stats.distributions import DistributionFactory
from grail.graph.base import VariableNode
from grail.logger import logger

class Engine:
    """
    The Engine converts a GRAIL Frame into an executable Pyro model.
    """
    
    def __init__(self, frame: Frame):
        self.frame = frame
        logger.info(f"Engine initialized for Frame: {frame.name}")

    def get_model(self) -> Callable:
        """
        Returns a callable Pyro model function based on the Frame.
        """
        graph = self.frame.graph
        topo_order = graph.topological_sort()
        logger.debug(f"Topological order for execution: {topo_order}")
        
        # We need to capture the graph structure in the closure
        def model(data: Dict[str, Any] = None):
            if data is None:
                data = {}
                
            runtime_values = {} # Local storage for sampled values in this trace
            
            for node_id in topo_order:
                node = graph.get_node(node_id)
                if not isinstance(node, VariableNode):
                    continue
                
                # Check if this variable has parents
                parents = graph.get_parents(node_id)
                
                # Resolve parameters dynamically
                # This is a simplfied logic:
                # 1. We copy the static params
                resolved_params = node.distribution_params.copy()
                
                # 2. If we have parents, we might need to use their values.
                # For this v0.X, let's assume if 'loc' (or similar) is a string matching a parent ID,
                # we substitute the parent's value.
                # Realistically, we'd need a more robust functional definition (e.g. loc = 2*parent + 3)
                
                for param_name, param_val in resolved_params.items():
                    if isinstance(param_val, str) and param_val in runtime_values:
                        resolved_params[param_name] = runtime_values[param_val]
                        
                # Get the distribution
                dist = DistributionFactory.get_distribution(node.distribution_name, resolved_params)
                
                # Check for observed data
                obs = None
                if node.is_observed:
                    if node.observations is not None:
                        obs = torch.tensor(node.observations)
                    elif node_id in data:
                        obs = torch.tensor(data[node_id])
                
                # Sample
                val = pyro.sample(node.name, dist, obs=obs)
                runtime_values[node_id] = val
                
            return runtime_values
            
        return model
