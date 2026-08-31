import pytest

from grail.stats.distributions import DistributionFactory


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


