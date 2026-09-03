"""Variable construction, prior normalization, and node binding."""


import pytest

from grail.frame.variable import DistributionPrior, Variable


def test_prior_defaults_to_an_unconfigured_distribution():
    variable = Variable(name="Bare")

    assert variable.get_distribution_name() is None
    assert variable.get_distribution_params() == {}


def test_a_mapping_prior_is_normalized_into_a_distribution_prior():
    variable = Variable(
        name="Mapped", prior={"distribution": "normal", "params": {"loc": 1.0}}
    )

    assert isinstance(variable.prior, DistributionPrior)
    assert variable.get_distribution_name() == "normal"
    assert variable.get_distribution_params() == {"loc": 1.0}


def test_prior_params_are_copied_rather_than_aliased():
    params = {"loc": 1.0}
    variable = Variable(name="Copied", prior=DistributionPrior("normal", params))

    params["loc"] = 99.0

    assert variable.get_distribution_params() == {"loc": 1.0}


def test_get_distribution_params_returns_a_defensive_copy():
    variable = Variable(name="Guarded")
    variable.set_distribution("normal", {"loc": 0.0, "scale": 1.0})

    variable.get_distribution_params()["loc"] = 42.0

    assert variable.get_distribution_params()["loc"] == 0.0


def test_invalid_prior_shapes_are_rejected():
    with pytest.raises(TypeError, match="must be a DistributionPrior or mapping"):
        Variable(name="Bad", prior="normal")

    with pytest.raises(TypeError, match=r"prior\['distribution'\] must be a string"):
        Variable(name="Bad", prior={"distribution": 5})

    with pytest.raises(TypeError, match=r"prior\['params'\] must be a mapping"):
        Variable(name="Bad", prior={"distribution": "normal", "params": [1, 2]})


def test_observations_round_trip_and_report_observed_status():
    variable = Variable(name="Observed")
    assert not variable.is_observed()

    variable.set_observations([1, 0, 1])
    assert variable.is_observed()
    assert variable.get_observations() == [1, 0, 1]

    variable.clear_observations()
    assert not variable.is_observed()
    assert variable.get_observations() is None


def test_distribution_spec_is_serializable():
    variable = Variable(name="Spec")
    variable.set_distribution("beta", {"alpha": 2.0, "beta": 3.0})

    assert variable.get_distribution_spec() == {
        "name": "beta",
        "params": {"alpha": 2.0, "beta": 3.0},
    }


def test_node_binding_is_idempotent_but_not_transferable():
    variable = Variable(name="Bound")

    variable.bind_node_id("node-1")
    variable.bind_node_id("node-1")
    assert variable.node_id == "node-1"

    with pytest.raises(ValueError, match="already bound to node id 'node-1'"):
        variable.bind_node_id("node-2")
