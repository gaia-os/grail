"""End-to-end Frame -> Engine -> Runner simulation behaviour."""


import pytest
import torch

from grail.engine import Engine
from grail.frame import Frame
from grail.runner import Runner


def test_a_discrete_parent_shifts_its_continuous_child():
    frame = Frame("mixture")
    cause_id = frame.add_variable("A", "bernoulli", {"theta": 0.5})
    frame.add_variable("B", "normal", {"loc": cause_id, "scale": 0.1})
    frame.add_dependency("A", "B")

    samples = Runner(Engine(frame).get_model()).simulate(num_samples=400)

    assert samples["A"].shape == (400,)
    assert samples["B"].shape == (400,)
    # B ~ Normal(A, 0.1), so B clusters at 0 when A is 0 and at 1 when A is 1.
    assert float(samples["B"][samples["A"] == 0.0].mean()) == pytest.approx(0.0, abs=0.1)
    assert float(samples["B"][samples["A"] == 1.0].mean()) == pytest.approx(1.0, abs=0.1)


def test_an_intervention_overrides_the_natural_cause():
    frame = Frame("causal-test")
    cause_id = frame.add_variable("X", "normal", {"loc": 0.0, "scale": 1.0})
    frame.add_variable("Y", "normal", {"loc": cause_id, "scale": 0.1})
    frame.add_dependency("X", "Y")
    runner = Runner(Engine(frame).get_model())

    observational = runner.simulate(num_samples=200)
    interventional = runner.do_operation({"X": torch.tensor(10.0)}, num_samples=200)

    assert float(observational["Y"].mean()) == pytest.approx(0.0, abs=0.3)
    assert float(interventional["Y"].mean()) == pytest.approx(10.0, abs=0.1)


def test_a_constant_variable_drives_its_dependent():
    frame = Frame("constant-parent")
    bias_id = frame.add_variable("Bias", "constant", {"value": 2.5})
    frame.add_variable("Y", "normal", {"loc": bias_id, "scale": 0.05})
    frame.add_dependency("Bias", "Y")

    samples = Runner(Engine(frame).get_model()).simulate(num_samples=150)

    assert torch.allclose(samples["Bias"], torch.full_like(samples["Bias"], 2.5))
    assert float(samples["Y"].mean()) == pytest.approx(2.5, abs=0.05)


def test_non_canonical_distribution_parameters_fail_at_compile_time():
    frame = Frame("strict-params")
    frame.add_variable("A", "bernoulli", {"p": 0.9})

    with pytest.raises(ValueError, match="does not accept params"):
        Runner(Engine(frame).get_model()).simulate(num_samples=5)


def test_a_deep_chain_propagates_through_every_level():
    frame = Frame("deep-chain")
    previous = frame.add_variable("Level0", "constant", {"value": 1.0})
    for level in range(1, 4):
        current = frame.add_variable(
            f"Level{level}", "normal", {"loc": previous, "scale": 0.01}
        )
        frame.add_dependency(f"Level{level - 1}", f"Level{level}")
        previous = current

    samples = Runner(Engine(frame).get_model()).simulate(num_samples=200)

    assert float(samples["Level3"].mean()) == pytest.approx(1.0, abs=0.05)


def test_conditioning_on_a_child_shifts_the_inferred_parent():
    """Observing an effect should move the fitted belief about its cause."""
    frame = Frame("conditioned")
    cause_id = frame.add_variable("Cause", "normal", {"loc": 0.0, "scale": 1.0})
    frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 0.1})
    frame.add_dependency("Cause", "Effect")
    runner = Runner(Engine(frame).get_model())

    runner.train_svi(data={"Effect": torch.tensor(3.0)}, n_steps=800, learning_rate=0.05)
    posterior = runner.predict(num_samples=400)

    assert float(posterior["Cause"].mean()) == pytest.approx(3.0, abs=0.4)
