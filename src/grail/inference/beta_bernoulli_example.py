"""Exact, resumable inference for Beta priors with Bernoulli likelihoods."""

from typing import TYPE_CHECKING, Any

from grail.frame.state import PosteriorState
from grail.inference.base import InferenceStrategy

if TYPE_CHECKING:
    from grail.frame import Frame
    from grail.frame.variable import Variable


class BetaBernoulliInference(InferenceStrategy):
    """Perform conjugate updates without replaying previously processed batches.

    A compatible subgraph has a latent ``beta`` variable and one or more child
    ``bernoulli`` variables whose ``theta`` parameter references that latent
    variable.  Each new 0/1 child-observation batch increments the saved Beta
    parameters exactly once.  This strategy is intentionally independent of
    :class:`grail.engine.Engine` and can be replaced by another strategy for a
    different model family.

    Saved metadata records provenance for the update:

    ``processed_batch_ids``
        Every batch ever folded into this posterior.  Cumulative, and the field
        that makes a repeated :meth:`infer` a no-op rather than a double count.
    ``new_batch_ids``
        The batches consumed by the update that last *changed* the posterior.
        Inferring again with no new evidence returns the stored snapshot
        unchanged, so this field still describes that earlier update rather
        than resetting to empty.
    """

    name = "beta-bernoulli-exact"

    def infer(self, frame: "Frame") -> dict[str, PosteriorState]:
        """Update all compatible Beta variables from newly recorded Bernoulli evidence."""
        posteriors: dict[str, PosteriorState] = {}
        for variable in frame.get_variables():
            if variable.get_distribution_name() != "beta":
                continue
            likelihood_variables = self._likelihood_variables(frame, variable)
            if not likelihood_variables:
                continue
            posterior = self._update_variable(frame, variable, likelihood_variables)
            if posterior is not None:
                posteriors[variable.name] = posterior
        return posteriors

    def _update_variable(
        self, frame: "Frame", variable: "Variable", likelihood_variables: list["Variable"]
    ) -> PosteriorState | None:
        prior_params = variable.get_distribution_params()
        alpha_prior = _positive_number(prior_params.get("alpha", 1.0), "alpha")
        beta_prior = _positive_number(prior_params.get("beta", 1.0), "beta")
        prior = {"distribution": "beta", "params": {"alpha": alpha_prior, "beta": beta_prior}}
        existing = frame.get_posterior(variable.name, strategy=self.name)

        if existing is None:
            alpha, beta = alpha_prior, beta_prior
            processed_batch_ids: set[str] = set()
            successes = 0
            failures = 0
        else:
            if existing.distribution != "beta":
                raise ValueError(
                    f"saved posterior for '{variable.name}' is not a Beta distribution"
                )
            alpha = _positive_number(existing.params.get("alpha"), "saved alpha")
            beta = _positive_number(existing.params.get("beta"), "saved beta")
            processed_batch_ids = set(existing.metadata.get("processed_batch_ids", []))
            successes = int(existing.metadata.get("successes", round(alpha - alpha_prior)))
            failures = int(existing.metadata.get("failures", round(beta - beta_prior)))

        new_batch_ids = []
        observed_variable_names = []
        for likelihood_variable in likelihood_variables:
            for batch in frame.get_observation_batches(likelihood_variable.name):
                if batch.id in processed_batch_ids:
                    continue
                batch_successes, batch_failures = _count_bernoulli_values(
                    batch.values, likelihood_variable.name, batch.id
                )
                alpha += batch_successes
                beta += batch_failures
                successes += batch_successes
                failures += batch_failures
                processed_batch_ids.add(batch.id)
                new_batch_ids.append(batch.id)
                observed_variable_names.append(likelihood_variable.name)

        if existing is not None and not new_batch_ids:
            return existing

        return frame.save_posterior(
            variable.name,
            strategy=self.name,
            distribution="beta",
            prior=prior,
            params={"alpha": alpha, "beta": beta},
            metadata={
                "processed_batch_ids": sorted(processed_batch_ids),
                "new_batch_ids": new_batch_ids,
                "observation_variables": sorted(set(observed_variable_names)),
                "successes": successes,
                "failures": failures,
                "observation_count": successes + failures,
            },
        )

    @staticmethod
    def _likelihood_variables(frame: "Frame", theta: "Variable") -> list["Variable"]:
        if theta.node_id is None:
            return []
        likelihoods = []
        for child in frame.graph.get_children(theta.node_id):
            variable = getattr(child, "variable", None)
            if variable is None or variable.get_distribution_name() != "bernoulli":
                continue
            if variable.get_distribution_params().get("theta") == theta.node_id:
                likelihoods.append(variable)
        return likelihoods


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Beta parameter '{name}' must be a positive number")
    return float(value)


def _count_bernoulli_values(values: list[Any], variable_name: str, batch_id: str) -> tuple[int, int]:
    successes = 0
    failures = 0
    for value in values:
        if isinstance(value, bool):
            value = int(value)
        if value == 1:
            successes += 1
        elif value == 0:
            failures += 1
        else:
            raise ValueError(
                f"observation batch '{batch_id}' for Bernoulli variable '{variable_name}' "
                f"contains {value!r}; expected only 0 or 1"
            )
    return successes, failures
