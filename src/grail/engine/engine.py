from typing import Any, Callable, Dict, Optional

import pyro
import torch

from grail.frame import Frame
from grail.graph.base import VariableNode
from grail.logger import logger
from grail.stats.distributions import DistributionFactory


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
        def model(data: Optional[Dict[str, Any]] = None):
            if data is None:
                data = {}

            runtime_values = {}  # Local storage for sampled values in this trace

            for node_id in topo_order:
                node = graph.get_node(node_id)
                if not isinstance(node, VariableNode):
                    continue

                resolved_params = node.get_distribution_params()

                for param_name, param_val in resolved_params.items():
                    if isinstance(param_val, str) and param_val in runtime_values:
                        resolved_params[param_name] = runtime_values[param_val]

                distribution_name = node.get_distribution_name()
                if distribution_name is None:
                    raise ValueError(f"Variable '{node.name}' is missing a distribution specification")

                dist = DistributionFactory.get_distribution(distribution_name, resolved_params)

                obs = data.get(node.name, data.get(node_id))
                if obs is None and node.is_observed:
                    observations = node.get_observations()
                    if observations is not None:
                        obs = torch.tensor(observations)
                if obs is not None and not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs)

                val = pyro.sample(node.name, dist, obs=obs)
                runtime_values[node_id] = val

            return runtime_values

        return model
