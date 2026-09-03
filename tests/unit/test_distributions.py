import pytest
import torch

from grail.stats.distributions import Distribution, DistributionFactory


def test_factory_creates_distribution_by_code():
    bernoulli = DistributionFactory.create("bernoulli", {"theta": 0.8})

    assert float(bernoulli.mean) == pytest.approx(0.8)


def test_factory_rejects_unknown_param_names():
    with pytest.raises(ValueError, match="does not accept params"):
        DistributionFactory.create("normal", {"loc": 0.0, "scale": 1.0, "oops": 1})


def test_factory_rejects_non_canonical_bernoulli_params():
    with pytest.raises(ValueError, match="does not accept params"):
        DistributionFactory.create("bernoulli", {"p": 0.2})


def test_factory_rejects_non_lowercase_code():
    with pytest.raises(ValueError, match="Unknown distribution code"):
        DistributionFactory.create("Bernoulli", {"theta": 0.2})


def test_factory_creates_lognormal_distribution_by_code():
    lognormal = DistributionFactory.create("lognormal", {"loc": 0.0, "scale": 0.5})

    assert float(lognormal.loc) == pytest.approx(0.0)
    assert float(lognormal.scale) == pytest.approx(0.5)


def test_factory_creates_binomial_distribution_by_code():
    binomial = DistributionFactory.create("binomial", {"n": 10, "theta": 0.3})

    assert float(binomial.total_count) == pytest.approx(10.0)
    assert float(binomial.mean) == pytest.approx(3.0)


def test_factory_creates_beta_distribution_by_code():
    beta = DistributionFactory.create("beta", {"alpha": 2.0, "beta": 3.0})

    assert float(beta.concentration1) == pytest.approx(2.0)
    assert float(beta.concentration0) == pytest.approx(3.0)
    assert float(beta.mean) == pytest.approx(0.4)


def test_factory_rejects_unknown_binomial_param_names():
    with pytest.raises(ValueError, match="does not accept params"):
        DistributionFactory.create("binomial", {"n": 10, "rate": 0.3})


def test_factory_rejects_non_canonical_binomial_params():
    with pytest.raises(ValueError, match="does not accept params"):
        DistributionFactory.create("binomial", {"total_count": 10, "p": 0.3})


def test_factory_creates_constant_distribution_by_code():
    constant = DistributionFactory.create("constant", {"value": 42})

    assert float(constant.mean) == pytest.approx(42.0)


def test_factory_constant_requires_value_param():
    with pytest.raises(ValueError, match="requires param: value"):
        DistributionFactory.create("constant", {})


@pytest.mark.parametrize(
    ("code", "params"),
    [
        ("normal", {"loc": 0.0, "scale": 1.0}),
        ("bernoulli", {"theta": 0.5}),
        ("uniform", {"low": 0.0, "high": 1.0}),
        ("exponential", {"rate": 1.0}),
        ("gamma", {"concentration": 2.0, "rate": 1.0}),
        ("lognormal", {"loc": 0.0, "scale": 1.0}),
        ("binomial", {"n": 5, "theta": 0.5}),
        ("beta", {"alpha": 2.0, "beta": 2.0}),
        ("constant", {"value": 1.0}),
    ],
)
def test_every_registered_distribution_can_sample(code: str, params: dict):
    assert DistributionFactory.create(code, params).sample().numel() == 1


def test_get_distribution_is_an_alias_for_create():
    assert float(DistributionFactory.get_distribution("beta", {"alpha": 2.0, "beta": 2.0}).mean) == (
        pytest.approx(0.5)
    )


def test_to_tensor_converts_python_numerics_to_float32():
    for value in (1, 1.0, True, [1, 2]):
        assert Distribution.to_tensor(value).dtype == torch.float32


def test_to_tensor_passes_existing_tensors_through_unchanged():
    supplied = torch.tensor([1.0], dtype=torch.float64)

    assert Distribution.to_tensor(supplied) is supplied


def test_to_tensor_stacks_sequences_containing_tensors():
    """Resolved parameter references arrive as tensors that must keep batch dims."""
    stacked = Distribution.to_tensor([torch.tensor(1.0), 2.0])
    assert stacked.tolist() == [1.0, 2.0]

    batched = Distribution.to_tensor([torch.zeros(4), torch.ones(4)])
    assert batched.shape == (4, 2)


@pytest.mark.parametrize("value", ["Cause", {"$ref": "Cause"}])
def test_to_tensor_rejects_unresolved_reference_shapes(value):
    with pytest.raises(TypeError, match="denote variable references"):
        Distribution.to_tensor(value)
