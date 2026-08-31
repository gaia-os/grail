from typing import Any, Callable, Dict, Optional

import pyro
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim.optim import ClippedAdam
from pyro.infer.autoguide import AutoDiagonalNormal

from grail.logger import logger

class Runner:
    """
    Executes simulations and inference on GRAIL models.
    """

    def __init__(self, model: Callable):
        self.model = model
        self.guide = None  # AutoGuide will be generated if not provided
        logger.info("Runner initialized.")

    def simulate(self, num_samples: int = 1, data: Optional[Dict[str, Any]] = None):
        """
        Runs forward simulations (Prior Predictive) to generate data.
        """
        logger.info(f"Running simulation with {num_samples} samples.")
        predictive = Predictive(self.model, num_samples=num_samples)
        return predictive(data)

    def train_svi(self, data: Optional[Dict[str, Any]] = None, n_steps: int = 1000, learning_rate: float = 0.01):
        """
        Fits the model to data using Stochastic Variational Inference (SVI).
        Uses an AutoDiagonalNormal guide by default.
        """
        logger.info(f"Starting SVI training for {n_steps} steps, lr={learning_rate}")
        pyro.clear_param_store()

        # Simple AutoGuide
        self.guide = AutoDiagonalNormal(self.model)

        optimizer = ClippedAdam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        loss_history = []
        for i in range(n_steps):
            loss = svi.step(data)
            loss_history.append(loss)
            if i % 100 == 0:
                logger.info(f"[Step {i}] Loss: {loss}")

        return loss_history

    def predict(self, num_samples: int = 100, data: Optional[Dict[str, Any]] = None):
        """
        Posterior Predictive sampling after training.
        """
        if self.guide is None:
            raise ValueError("Model has not been trained (no guide found). Run train_svi first.")

        logger.info(f"Generating posterior predictions with {num_samples} samples.")
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        return predictive(data)

    def do_operation(self, interventions: Dict[str, Any], num_samples: int = 1):
        """
        Performs a 'do' operation (intervention) on the model.
        interventions: Dict mapping variable names to their fixed values.

        NOTE: Returned prediction does not "insert" the intervened values
        into the traces, it will simply return a "natural" sample.
        In other words, your intervened variables will still be sampled, and not return your interv. values.
        """
        logger.info(f"Performing do-operation with interventions: {interventions}")
        intervened_model = pyro.do(self.model, data=interventions)
        predictive = Predictive(intervened_model, num_samples=num_samples)
        # Note: We pass None for data usually in counterfactuals/interventions unless mixing obs
        return predictive()
