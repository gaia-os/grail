"""Compilation of Frames into executable Pyro models."""


import pytest
import torch

from grail.engine import Engine
from grail.frame import Frame, FrameSpec
from grail.runner import Runner


def test_model_samples_every_variable_in_topological_order():
    frame = Frame("ordered")
    cause_id = frame.add_variable("Cause", "constant", {"value": 3.0})
    effect_id = frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 0.01})
    frame.add_dependency("Cause", "Effect")

    values = Engine(frame).get_model()()

    assert set(values) == {cause_id, effect_id}
    assert float(values[cause_id]) == pytest.approx(3.0)
    assert float(values[effect_id]) == pytest.approx(3.0, abs=0.1)


def test_model_exposes_its_sample_site_names():
    frame = Frame("sites")
    frame.add_variable("Alpha", "normal", {"loc": 0.0, "scale": 1.0})
    frame.add_variable("Beta", "normal", {"loc": 0.0, "scale": 1.0})

    assert Engine(frame).get_model().variable_names == frozenset({"Alpha", "Beta"})


def test_parameters_may_reference_a_parent_by_name_or_node_id():
    frame = Frame("reference-styles")
    cause_id = frame.add_variable("Cause", "constant", {"value": 2.0})
    by_id = frame.add_variable("ById", "normal", {"loc": cause_id, "scale": 0.01})
    by_name = frame.add_variable("ByName", "normal", {"loc": "Cause", "scale": 0.01})
    frame.add_dependency("Cause", "ById")
    frame.add_dependency("Cause", "ByName")

    values = Engine(frame).get_model()()

    assert float(values[by_id]) == pytest.approx(2.0, abs=0.1)
    assert float(values[by_name]) == pytest.approx(2.0, abs=0.1)


def test_references_nested_in_list_parameters_are_resolved():
    """A vector parameter built from two parents must resolve element-wise."""
    spec = FrameSpec.model_validate(
        {
            "version": 1,
            "name": "vector-parameter",
            "variables": [
                {"name": "Low", "distribution": "constant", "params": {"value": -5.0}},
                {"name": "High", "distribution": "constant", "params": {"value": 5.0}},
                {
                    "name": "Pair",
                    "distribution": "normal",
                    "params": {
                        "loc": [{"$ref": "Low"}, {"$ref": "High"}],
                        "scale": 0.01,
                    },
                },
            ],
            "dependencies": [
                {"source": "Low", "target": "Pair"},
                {"source": "High", "target": "Pair"},
            ],
        }
    )
    frame = Frame.from_spec(spec)

    samples = Runner(Engine(frame).get_model()).simulate(num_samples=16)

    assert samples["Pair"].shape == (16, 2)
    assert samples["Pair"].mean(dim=0).tolist() == pytest.approx([-5.0, 5.0], abs=0.1)


def test_unknown_parameter_reference_is_reported_clearly():
    frame = Frame("dangling")
    frame.add_variable("Effect", "normal", {"loc": "Nope", "scale": 1.0})

    with pytest.raises(ValueError, match="references unknown variable 'Nope'"):
        Engine(frame).get_model()()


def test_reference_to_an_unsampled_parent_suggests_the_missing_dependency():
    """A backwards dependency edge orders the parent after its own child."""
    frame = Frame("misordered")
    cause_id = frame.add_variable("Cause", "normal", {"loc": 0.0, "scale": 1.0})
    frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 1.0})
    # The edge points the wrong way, so Effect is sampled before Cause.
    frame.add_dependency("Effect", "Cause")

    with pytest.raises(ValueError, match="has not been sampled yet"):
        Engine(frame).get_model()()


def test_unresolved_declarative_reference_names_the_compilation_step():
    frame = Frame("raw-ref")
    frame.add_variable("Cause", "normal", {"loc": 0.0, "scale": 1.0})
    frame.add_variable("Effect", "normal", {"loc": {"$ref": "Cause"}, "scale": 1.0})

    with pytest.raises(ValueError, match=r"unresolved \{'\$ref': 'Cause'\}"):
        Engine(frame).get_model()()


def test_missing_distribution_is_rejected():
    frame = Frame("no-distribution")
    frame.add_variable("Orphan", "normal", {"loc": 0.0, "scale": 1.0})
    frame.get_variable("Orphan").prior.distribution = None

    with pytest.raises(ValueError, match="missing a distribution specification"):
        Engine(frame).get_model()()


def test_frame_without_variables_cannot_compile():
    with pytest.raises(ValueError, match="no variables to compile"):
        Engine(Frame("empty")).get_model()


def test_observation_keys_must_name_a_variable():
    frame = Frame("typo-guard")
    frame.add_variable("Exercise", "bernoulli", {"theta": 0.5})
    model = Engine(frame).get_model()

    with pytest.raises(KeyError, match=r"no variables named \['Exercse'\]"):
        model(observations={"Exercse": torch.tensor(1.0)})


def test_observations_may_be_keyed_by_name_or_node_id():
    frame = Frame("keying")
    node_id = frame.add_variable("Outcome", "normal", {"loc": 0.0, "scale": 1.0})
    model = Engine(frame).get_model()

    assert float(model({"Outcome": torch.tensor(7.0)})[node_id]) == pytest.approx(7.0)
    assert float(model({node_id: torch.tensor(9.0)})[node_id]) == pytest.approx(9.0)


def test_runtime_observations_override_observations_attached_to_the_frame():
    frame = Frame("override")
    node_id = frame.add_variable(
        "Outcome", "normal", {"loc": 0.0, "scale": 1.0}, observations=[1.0]
    )
    model = Engine(frame).get_model()

    assert model()[node_id].tolist() == [1.0]
    assert model({"Outcome": torch.tensor([4.0])})[node_id].tolist() == [4.0]


def test_observations_are_coerced_to_float_tensors():
    """Integer YAML observations must not reach Pyro as an int64 tensor."""
    frame = Frame("dtype")
    node_id = frame.add_variable(
        "Toss", "bernoulli", {"theta": 0.5}, observations=[1, 1, 0]
    )

    observed = Engine(frame).get_model()()[node_id]

    assert observed.dtype == torch.float32
    assert observed.tolist() == [1.0, 1.0, 0.0]


def test_scalar_runtime_observation_is_coerced_to_a_float32_scalar_tensor():
    frame = Frame("runtime-scalar-dtype")
    node_id = frame.add_variable("Outcome", "normal", {"loc": 0.0, "scale": 1.0})

    observed = Engine(frame).get_model()({"Outcome": 7})[node_id]

    assert observed.dtype == torch.float32
    assert observed.shape == torch.Size()
    assert observed.item() == pytest.approx(7.0)


def test_existing_tensors_are_passed_through_unchanged():
    frame = Frame("passthrough")
    node_id = frame.add_variable("Count", "normal", {"loc": 0.0, "scale": 1.0})
    supplied = torch.tensor([2.0, 3.0], dtype=torch.float64)

    assert Engine(frame).get_model()({"Count": supplied})[node_id] is supplied
