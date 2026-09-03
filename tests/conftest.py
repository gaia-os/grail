"""Shared fixtures for the GRAIL test suite.

Tests are split into ``unit/`` (one module or class in isolation) and
``integration/`` (Frame -> Engine -> Runner -> FrameState working together).
"""


from pathlib import Path

import pyro
import pytest

from grail.frame import Frame, FrameRepository

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A minimal conjugate model reused across state, inference, and workflow tests.
COIN_MODEL_YAML = """
version: 1
name: coin-model
variables:
  - name: Theta
    distribution: beta
    params:
      alpha: 1.0
      beta: 1.0
  - name: Toss
    distribution: bernoulli
    params:
      theta:
        $ref: Theta
dependencies:
  - source: Theta
    target: Toss
""".lstrip()


@pytest.fixture(autouse=True)
def isolate_pyro_state():
    """
    Isolate every test from Pyro's global parameter store and RNG.

    ``Runner.train_svi`` writes into a process-wide store, so without this a
    test's guide parameters would leak into whichever test ran next.
    """
    pyro.clear_param_store()
    pyro.set_rng_seed(20260903)
    yield
    pyro.clear_param_store()


@pytest.fixture
def project_root() -> Path:
    """The repository root, for tests that read committed example data."""
    return PROJECT_ROOT


@pytest.fixture
def repository(tmp_path: Path) -> FrameRepository:
    """A FrameRepository rooted in an empty temporary directory."""
    return FrameRepository(
        tmp_path / "frames", state_database_path=tmp_path / "state.sqlite3"
    )


@pytest.fixture
def coin_frame(repository: FrameRepository) -> Frame:
    """A Beta-Bernoulli Frame compiled from YAML and backed by a state store."""
    (repository.root / "coin-model.yaml").write_text(COIN_MODEL_YAML, encoding="utf-8")
    return repository.load("coin-model.yaml")


@pytest.fixture
def chain_frame() -> Frame:
    """An in-memory ``Cause -> Effect`` Frame with no persistence attached."""
    frame = Frame("chain-frame")
    cause_id = frame.add_variable("Cause", "normal", {"loc": 0.0, "scale": 1.0})
    frame.add_variable("Effect", "normal", {"loc": cause_id, "scale": 0.1})
    frame.add_dependency("Cause", "Effect")
    return frame
