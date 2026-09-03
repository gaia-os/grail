"""Exact conjugate updating for Beta priors with Bernoulli likelihoods."""


import pytest

from grail.frame import Frame, FrameRepository
from grail.inference import BetaBernoulliInference

STRATEGY = BetaBernoulliInference.name


def test_a_single_batch_increments_the_beta_parameters(coin_frame: Frame):
    coin_frame.record_observations("Toss", [1, 1, 1, 0], batch_id="b1")

    posterior = BetaBernoulliInference().infer(coin_frame)["Theta"]

    assert posterior.params == {"alpha": 4.0, "beta": 2.0}
    assert posterior.prior == {"distribution": "beta", "params": {"alpha": 1.0, "beta": 1.0}}
    assert posterior.metadata["successes"] == 3
    assert posterior.metadata["failures"] == 1
    assert posterior.metadata["observation_count"] == 4


def test_repeated_inference_does_not_recount_processed_batches(coin_frame: Frame):
    coin_frame.record_observations("Toss", [1, 1, 0], batch_id="b1")
    strategy = BetaBernoulliInference()

    first = strategy.infer(coin_frame)["Theta"]
    second = strategy.infer(coin_frame)["Theta"]

    assert first.params == second.params == {"alpha": 3.0, "beta": 2.0}
    assert second.metadata["processed_batch_ids"] == ["b1"]
    # No new evidence means the stored snapshot is returned untouched, so its
    # provenance still describes the update that last changed it.
    assert second.updated_at == first.updated_at
    assert second.metadata["new_batch_ids"] == ["b1"]


def test_a_later_batch_updates_incrementally(coin_frame: Frame):
    strategy = BetaBernoulliInference()
    coin_frame.record_observations("Toss", [1, 1], batch_id="b1")
    strategy.infer(coin_frame)

    coin_frame.record_observations("Toss", [0], batch_id="b2")
    posterior = strategy.infer(coin_frame)["Theta"]

    assert posterior.params == {"alpha": 3.0, "beta": 2.0}
    assert posterior.metadata["new_batch_ids"] == ["b2"]
    assert posterior.metadata["processed_batch_ids"] == ["b1", "b2"]


def test_boolean_observations_count_as_successes_and_failures(coin_frame: Frame):
    coin_frame.record_observations("Toss", [True, True, False], batch_id="b1")

    assert BetaBernoulliInference().infer(coin_frame)["Theta"].params == {
        "alpha": 3.0,
        "beta": 2.0,
    }


def test_non_binary_observations_are_rejected(coin_frame: Frame):
    coin_frame.record_observations("Toss", [1, 2], batch_id="b1")

    with pytest.raises(ValueError, match="expected only 0 or 1"):
        BetaBernoulliInference().infer(coin_frame)


def test_a_frame_without_a_bernoulli_child_yields_no_posterior(repository: FrameRepository):
    frame = Frame("lonely-beta")
    frame.add_variable("Theta", "beta", {"alpha": 1.0, "beta": 1.0})
    frame.attach_state_store(repository.state_store)

    assert BetaBernoulliInference().infer(frame) == {}


def test_a_child_that_does_not_reference_the_latent_is_ignored(repository: FrameRepository):
    """A Bernoulli child with a fixed rate is not evidence about Theta."""
    frame = Frame("unlinked")
    frame.add_variable("Theta", "beta", {"alpha": 1.0, "beta": 1.0})
    frame.add_variable("Toss", "bernoulli", {"theta": 0.5})
    frame.add_dependency("Theta", "Toss")
    frame.attach_state_store(repository.state_store)
    frame.record_observations("Toss", [1, 1, 1], batch_id="b1")

    assert BetaBernoulliInference().infer(frame) == {}


def test_non_positive_prior_parameters_are_rejected(repository: FrameRepository):
    frame = Frame("bad-prior")
    theta_id = frame.add_variable("Theta", "beta", {"alpha": 0.0, "beta": 1.0})
    frame.add_variable("Toss", "bernoulli", {"theta": theta_id})
    frame.add_dependency("Theta", "Toss")
    frame.attach_state_store(repository.state_store)
    frame.record_observations("Toss", [1], batch_id="b1")

    with pytest.raises(ValueError, match="must be a positive number"):
        BetaBernoulliInference().infer(frame)


def test_a_saved_posterior_of_another_family_is_rejected(coin_frame: Frame):
    coin_frame.save_posterior(
        "Theta",
        strategy=STRATEGY,
        distribution="normal",
        prior={"distribution": "beta", "params": {"alpha": 1.0, "beta": 1.0}},
        params={"loc": 0.0, "scale": 1.0},
        metadata={},
    )
    coin_frame.record_observations("Toss", [1], batch_id="b1")

    with pytest.raises(ValueError, match="is not a Beta distribution"):
        BetaBernoulliInference().infer(coin_frame)


def test_every_compatible_latent_in_a_frame_is_updated(repository: FrameRepository):
    frame = Frame("two-coins")
    for name in ("Alpha", "Beta"):
        theta_id = frame.add_variable(f"{name}Rate", "beta", {"alpha": 1.0, "beta": 1.0})
        frame.add_variable(f"{name}Toss", "bernoulli", {"theta": theta_id})
        frame.add_dependency(f"{name}Rate", f"{name}Toss")
    frame.attach_state_store(repository.state_store)
    frame.record_observations("AlphaToss", [1, 1], batch_id="a1")
    frame.record_observations("BetaToss", [0, 0, 0], batch_id="b1")

    posteriors = BetaBernoulliInference().infer(frame)

    assert posteriors["AlphaRate"].params == {"alpha": 3.0, "beta": 1.0}
    assert posteriors["BetaRate"].params == {"alpha": 1.0, "beta": 4.0}


def test_evidence_from_several_children_accumulates(repository: FrameRepository):
    frame = Frame("shared-rate")
    theta_id = frame.add_variable("Theta", "beta", {"alpha": 1.0, "beta": 1.0})
    for child in ("SiteA", "SiteB"):
        frame.add_variable(child, "bernoulli", {"theta": theta_id})
        frame.add_dependency("Theta", child)
    frame.attach_state_store(repository.state_store)
    frame.record_observations("SiteA", [1, 1], batch_id="a1")
    frame.record_observations("SiteB", [1, 0], batch_id="b1")

    posterior = BetaBernoulliInference().infer(frame)["Theta"]

    assert posterior.params == {"alpha": 4.0, "beta": 2.0}
    assert posterior.metadata["observation_variables"] == ["SiteA", "SiteB"]
