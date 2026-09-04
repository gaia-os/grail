"""Execution of compiled Frame models: simulation, training, and inference."""

from collections.abc import Callable, Mapping
from typing import Any

import pyro
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim.optim import ClippedAdam
import torch

from grail.frame import Frame
from grail.frame.state import PosteriorState
from grail.inference import InferenceStrategy
from grail.logger import logger


class Runner:
    """
    Executes simulations and inference on GRAIL models.

    The model is the callable produced by :meth:`grail.engine.Engine.get_model`.
    A Frame is only required for :meth:`infer`, which persists posterior state.
    """

    def __init__(self, model: Callable, *, frame: Frame | None = None):
        self.model = model
        self.frame = frame
        self.guide = None  # Set by train_svi; required by predict.
        logger.info("Runner initialized.")

    def simulate(
        self, num_samples: int = 1, observations: Mapping[str, Any] | None = None
    ):
        """
        Runs forward simulations (Prior Predictive) to generate data.
        """
        logger.info(f"Running simulation with {num_samples} samples.")
        predictive = Predictive(self.model, num_samples=num_samples)
        return predictive(observations)

    def train_svi(
        self,
        observations: Mapping[str, Any] | None = None,
        n_steps: int = 1000,
        learning_rate: float = 0.01,
        guide: Callable | None = None,
    ):
        """
        Fits the model to observations using Stochastic Variational Inference (SVI).

        Uses an ``AutoDiagonalNormal`` guide unless one is supplied. This clears
        Pyro's global parameter store so each fit starts fresh, and leaves the
        trained guide on the Runner for :meth:`predict`. It does not persist a
        resumable posterior on the Frame; :meth:`infer` does that.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")

        logger.info(f"Starting SVI training for {n_steps} steps, lr={learning_rate}")
        pyro.clear_param_store()

        self.guide = guide if guide is not None else AutoDiagonalNormal(self.model)

        optimizer = ClippedAdam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        log_interval = max(1, n_steps // 10)
        loss_history = []
        for step in range(n_steps):
            loss = svi.step(observations)
            loss_history.append(loss)
            if step % log_interval == 0:
                logger.info(f"[Step {step}] Loss: {loss}")

        logger.info(f"SVI training complete. Final loss: {loss_history[-1]}")
        return loss_history

    def infer(self, strategy: InferenceStrategy) -> dict[str, PosteriorState]:
        """
        Update the Frame using its saved observations.

        This turns saved observations into a saved posterior: the model's latest
        learned state. The strategy chooses how to perform that update. The
        existing ``train_svi`` method is separate: it starts a fresh, temporary
        analysis and does not save a resumable posterior on the Frame.
        """
        if self.frame is None:
            raise ValueError(
                "Runner.infer requires a Frame. Construct Runner(model, frame=frame)."
            )
        logger.info("Running inference strategy '%s' for Frame '%s'", strategy.name, self.frame.name)
        return strategy.infer(self.frame)

    def predict(self, num_samples: int = 100):
        """
        Posterior Predictive sampling after training.
        """
        if self.guide is None:
            raise ValueError("Model has not been trained (no guide found). Run train_svi first.")

        logger.info(f"Generating posterior predictions with {num_samples} samples.")
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        return predictive()

    def do_operation(
        self,
        interventions: Mapping[str, Any],
        num_samples: int = 1,
        observations: Mapping[str, Any] | None = None,
    ):
        """
        Performs a 'do' operation (intervention) on the model.

        ``interventions`` maps variable names to the values they are fixed to.

        Pyro's ``do`` implements a Single World Intervention Graph: downstream
        variables see the intervened value, but the trace's own site for an
        intervened variable holds a fresh draw from its original distribution
        that propagates nowhere. Returning that draw would misreport the world
        the caller asked about, so the intervened sites are replaced with the
        values actually imposed.
        """
        if not interventions:
            raise ValueError("do_operation requires at least one intervention")
        self._validate_site_names(interventions, "intervention")

        logger.info(f"Performing do-operation with interventions: {dict(interventions)}")
        intervened_model = pyro.do(self.model, data=dict(interventions))
        predictive = Predictive(intervened_model, num_samples=num_samples)
        samples = predictive(observations)

        for name, value in interventions.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.tensor(
                value, dtype=torch.float32
            )
            samples[name] = tensor.expand(torch.Size([num_samples]) + tensor.shape).clone()
        return samples

    def _validate_site_names(self, names: Mapping[str, Any], role: str) -> None:
        """Reject names that the compiled model does not sample, when knowable."""
        variable_names = getattr(self.model, "variable_names", None)
        if variable_names is None:
            # A hand-written model callable does not advertise its sites.
            return
        unknown = sorted(set(names) - set(variable_names))
        if unknown:
            raise KeyError(
                f"{role} targets {unknown} which the model does not sample. "
                f"Known variables: {sorted(variable_names)}"
            )
