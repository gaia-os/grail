"""Simulation, training, intervention, and inference dispatch on compiled models."""


import pytest
import torch

from grail.engine import Engine
from grail.frame import Frame
from grail.inference import BetaBernoulliInference
from grail.runner import Runner


@pytest.fixture
def chain_runner(chain_frame: Frame) -> Runner:
    return Runner(Engine(chain_frame).get_model(), frame=chain_frame)


def test_simulate_draws_every_site_with_the_requested_sample_count(chain_runner: Runner):
    samples = chain_runner.simulate(num_samples=32)

    assert set(samples) == {"Cause", "Effect"}
    assert samples["Cause"].shape == (32,)
    assert samples["Effect"].shape == (32,)


def test_simulate_propagates_parent_values_to_children(chain_runner: Runner):
    samples = chain_runner.simulate(num_samples=256)

    # Effect ~ Normal(Cause, 0.1), so the two should track each other closely.
    assert torch.abs(samples["Effect"] - samples["Cause"]).mean() < 0.2


def test_train_svi_recovers_a_known_location(chain_frame: Frame):
    """SVI on Cause given a tightly-coupled observed Effect should find its value."""
    runner = Runner(Engine(chain_frame).get_model(), frame=chain_frame)

    losses = runner.train_svi(
        data={"Effect": torch.tensor(2.0)}, n_steps=600, learning_rate=0.05
    )

    assert len(losses) == 600
    assert losses[-1] < losses[0]
    posterior_cause = runner.predict(num_samples=400)["Cause"]
    assert float(posterior_cause.mean()) == pytest.approx(2.0, abs=0.35)


def test_train_svi_rejects_a_non_positive_step_count(chain_runner: Runner):
    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        chain_runner.train_svi(n_steps=0)


def test_train_svi_accepts_a_caller_supplied_guide(chain_frame: Frame):
    from pyro.infer.autoguide import AutoNormal

    runner = Runner(Engine(chain_frame).get_model())
    guide = AutoNormal(runner.model)

    runner.train_svi(data={"Effect": torch.tensor(1.0)}, n_steps=10, guide=guide)

    assert runner.guide is guide


def test_predict_requires_training_first(chain_runner: Runner):
    with pytest.raises(ValueError, match="has not been trained"):
        chain_runner.predict(num_samples=4)


def test_do_operation_reports_the_value_actually_imposed(chain_runner: Runner):
    """Pyro's SWIG semantics leave a non-propagating draw at the intervened site."""
    samples = chain_runner.do_operation({"Cause": torch.tensor(10.0)}, num_samples=64)

    assert samples["Cause"].shape == (64,)
    assert torch.allclose(samples["Cause"], torch.full((64,), 10.0))


def test_do_operation_propagates_the_intervention_downstream(chain_runner: Runner):
    samples = chain_runner.do_operation({"Cause": torch.tensor(10.0)}, num_samples=64)

    assert float(samples["Effect"].mean()) == pytest.approx(10.0, abs=0.1)


def test_do_operation_accepts_plain_python_numbers(chain_runner: Runner):
    samples = chain_runner.do_operation({"Cause": 4.0}, num_samples=8)

    assert torch.allclose(samples["Cause"], torch.full((8,), 4.0))
    assert float(samples["Effect"].mean()) == pytest.approx(4.0, abs=0.15)


def test_do_operation_requires_at_least_one_intervention(chain_runner: Runner):
    with pytest.raises(ValueError, match="at least one intervention"):
        chain_runner.do_operation({}, num_samples=4)


def test_do_operation_rejects_a_variable_the_model_does_not_sample(chain_runner: Runner):
    with pytest.raises(KeyError, match=r"intervention targets \['Casue'\]"):
        chain_runner.do_operation({"Casue": 1.0}, num_samples=4)


def test_do_operation_tolerates_a_hand_written_model():
    """A model that is not Engine-compiled cannot advertise its sites, so skip validation."""
    import pyro
    import pyro.distributions as dist

    def model(data=None):
        return {"X": pyro.sample("X", dist.Normal(0.0, 1.0))}

    samples = Runner(model).do_operation({"X": 3.0}, num_samples=5)

    assert torch.allclose(samples["X"], torch.full((5,), 3.0))


def test_infer_requires_a_frame(chain_frame: Frame):
    runner = Runner(Engine(chain_frame).get_model())

    with pytest.raises(ValueError, match="requires a Frame"):
        runner.infer(BetaBernoulliInference())
